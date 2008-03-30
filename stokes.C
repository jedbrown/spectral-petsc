static char help[] = "Stokes problem with non-Newtonian rheology via Chebyshev collocation.\n";

//#define InFunction printf("%s\n", __FUNCT__)
#define InFunction
#define SOLVE 1
#define CHECK_EXACT 1

#include "chebyshev.h"
#include "util.C"
#include <petscsnes.h>
#include <stdbool.h>

typedef enum { DIRICHLET, NEUMANN } BdyType;

typedef PetscErrorCode(*ExactSolution)(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
typedef PetscErrorCode(*BdyFunc)(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);
typedef PetscErrorCode(*Rheology)(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta, void *ctx);

typedef struct {
  PetscInt       debug;
  PetscReal      hardness, exponent, regularization;
  Rheology       rheology;
  ExactSolution  exact;
  BdyFunc        boundary;
  void          *rheologyCtx, *exactCtx, *boundaryCtx;
} StokesOptions;

typedef struct {
  MPI_Comm       comm;
  StokesOptions *options;
  PetscInt       numFields, numDims, numWork, numWorkScalar, numWorkVel;
  PetscInt      *dim;
  KSP            KSPSchur, KSPVelocity;
  Mat            MatSchur, MatPV, MatVP, MatVV, MatVVPC; // Used by preconditioner
  Mat           *Dscalar;       // derivative matrices, operates on scalar valued local Vec, used by preconditioner
  Mat           *Dfield;        // derivative matrices, operates on vector valued local Vec
  Vec           *workScalar;    // used by preconditioner
  Vec           *workVel;       // used by preconditioner, numDims
  Vec           *work, *gradu;  // each is vector valued, numFields
  Vec            eta, deta;     // effective viscosity and it's derivative w.r.t second invariant, scalar valued
  Vec            coord;         // vector valued, numDims
  Vec            xL;            // vector valued, numFields
  Vec            pressureG, velocityG;
  Vec            dirichlet, dirichletP, dirichletV;     // special dirichlet format
  Vec            force;         // Global body force vector
  IS             isLG, isGL;    // vector valued local <-> global
  VecScatter     scatterLG, scatterGL;
  IS             isDL;
  VecScatter     scatterLD, scatterDL; // vector valued local <-> special dirichlet
  VecScatter     scatterPG, scatterGP; // scalar valued local <-> pressure global
  VecScatter     scatterVG, scatterGV; // velocity valued local <-> velocity global
  VecScatter     scatterPD, scatterDP; // local pressure <-> dirichlet pressure
  VecScatter     scatterVD, scatterDV; // local velocity <-> dirichlet velocity
} StokesCtx;

PetscErrorCode StokesCreate(MPI_Comm comm, Mat *A, Vec *x, StokesCtx **);
PetscErrorCode StokesDestroy(StokesCtx *);
PetscErrorCode StokesProcessOptions(StokesCtx *);
PetscErrorCode StokesMatMult(Mat, Vec, Vec);
PetscErrorCode StokesMatMultSchur(Mat, Vec, Vec);
PetscErrorCode StokesMatMultPV(Mat, Vec, Vec);
PetscErrorCode StokesMatMultVP(Mat, Vec, Vec);
PetscErrorCode StokesMatMultVV(Mat, Vec, Vec);
PetscErrorCode StokesFunction(SNES, Vec, Vec, void *);
PetscErrorCode StokesJacobian(SNES, Vec, Mat *, Mat *, MatStructure *, void *);
PetscErrorCode StokesSetupDomain(StokesCtx *);
PetscErrorCode StokesCreateExactSolution(SNES, Vec u, Vec u2);
PetscErrorCode StokesCheckResidual(SNES snes, Vec u, Vec x);
PetscErrorCode StokesRemoveConstantPressure(KSP, StokesCtx *, Vec *, MatNullSpace *);
PetscErrorCode StokesPressureReduceOrder(Vec u, StokesCtx *c);
PetscErrorCode StokesPCApply(void *, Vec, Vec);
PetscErrorCode StokesSchurMatMult(Mat, Vec, Vec);
PetscErrorCode StokesVelocityMatMult(Mat, Vec, Vec);

//PetscErrorCode StokesRheologyPowerLaw(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta);
PetscErrorCode VecPrint2(Vec v, const char *name, StokesCtx *ctx);
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta, void *ctx);
PetscErrorCode StokesExactNull(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesExactCos(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesExactTest(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesDirichlet(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  MPI_Comm             comm = PETSC_COMM_SELF;
  StokesCtx           *ctx;
  SNES                 snes;
  KSP                  ksp;
  PC                   pc;
  MatNullSpace         ns;
  Mat                  A, P;
  Vec                  x, r, u, u2, nv;
  PetscReal            norm, rnorm, unorm, u2norm;
  PetscInt             its;
  SNESConvergedReason  reason;
  PetscTruth           flag;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
  ierr = fftw_import_system_wisdom();CHKERRQ(ierr);
  ierr = StokesCreate(comm, &A, &x, &ctx);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &r);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &u2);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes, ctx);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, StokesFunction, ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, A, StokesJacobian, ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPFGMRES);CHKERRQ(ierr);
  ierr = StokesRemoveConstantPressure(ksp, ctx, &nv, &ns);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc, ctx);CHKERRQ(ierr);
  ierr = PCShellSetApply(pc, StokesPCApply);CHKERRQ(ierr);
  // Normally, we would also need a setup function, but this work will be done in StokesFunction

  // u = exact solution, u2 = A(u) u (used as forcing term)
  ierr = StokesCreateExactSolution(snes, u, u2);CHKERRQ(ierr);
  ierr = StokesFunction(snes, u, r, ctx);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, &unorm);CHKERRQ(ierr);
  ierr = VecNorm(u2, NORM_INFINITY, &u2norm);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_INFINITY, &rnorm);CHKERRQ(ierr);
  {
    ierr = PetscPrintf(comm, "Norm of solution %9.3e  norm of forcing %9.3e  norm of residual %9.3e\n", unorm, u2norm, rnorm);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceTest(ns, A);CHKERRQ(ierr);

  if (true) {
    ierr = VecSet(x, 0.0);CHKERRQ(ierr);
    ierr = SNESSolve(snes, PETSC_NULL, x);CHKERRQ(ierr);

    ierr = VecAXPY(x, -1.0, u);CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_INFINITY, &norm);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of nonlinear iterations = %d\n", its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%-25s: abs = %8e\n", "Norm of error", norm);CHKERRQ(ierr);
  }

  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = StokesDestroy(ctx);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);           ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);           ierr = VecDestroy(u2);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(ns);CHKERRQ(ierr); ierr = VecDestroy(nv);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesCreate"
PetscErrorCode StokesCreate(MPI_Comm comm, Mat *A, Vec *X, StokesCtx **ctx)
{
  const unsigned  fftw_flag = FFTW_ESTIMATE;
  StokesCtx      *c;
  StokesOptions  *opt;
  PetscInt        d, m;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(1, StokesCtx, ctx, 1, StokesOptions, &opt);CHKERRQ(ierr);
  c = *ctx; c->options = opt; c->comm = comm;
  ierr = StokesProcessOptions(c);CHKERRQ(ierr);
  d = c->numDims;
  ierr = PetscMalloc(d*sizeof(Mat), &c->Dfield);CHKERRQ(ierr);
  m = productInt(d, c->dim); // number of local nodes
  c->numWork = 2*d+1; // very generous
  c->numWorkScalar = 2*d+1;
  c->numWorkVel = 2*d+1;
  ierr = VecCreate(comm, &c->eta);CHKERRQ(ierr); // Scalar valued
  ierr = VecSetSizes(c->eta, m, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c->eta);CHKERRQ(ierr);
  ierr = VecCreate(comm, &c->coord);CHKERRQ(ierr); // Vector valued, one component for each dimension
  ierr = VecSetSizes(c->coord, m*d, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->coord, d);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c->coord);CHKERRQ(ierr);
  ierr = VecCreate(comm, &c->xL);CHKERRQ(ierr); // Vector valued, one component for each dimension plus pressure
  ierr = VecSetSizes(c->xL, m*(d+1), PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->xL, d+1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c->xL);CHKERRQ(ierr);
  ierr = VecDuplicate(c->eta, c->deta);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->eta, c->numWorkScalar, &c->workScalar);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->coord, c->numWorkVel, &c->workVel);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->xL, c->numWork, &c->work);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->xL, d, &c->gradu);CHKERRQ(ierr);
  { // Create differentiation matrices
    PetscInt cheb_dim[d+1];
    for (int i=0; i < d; i++) cheb_dim[i] = c->dim[i];
    cheb_dim[d] = d+1;
    for (int i=0; i < d; i++) {
      ierr = MatCreateCheb(comm, d+1, i, cheb_dim, fftw_flag, c->work[0], c->work[1], &c->Dfield[i]);CHKERRQ(ierr);
      ierr = MatCreateCheb(comm, d, i, c->dim, fftw_flag, c->workScalar[0], c->workScalar[1], &c->Dfield[i]);CHKERRQ(ierr);
    }
  }
  { // Set up coordinate mapping
    PetscReal *x;
    ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);
    for (BlockIt it = BlockIt(d, c->dim); !it.done; it.next()) {
      for (int j=0; j < d; j++) {
        x[it.i*d+j] = cos(it.ind[j] * PETSC_PI / (c->dim[j] - 1));
      }
    }
    ierr = VecRestoreArray(c->coord, &x);CHKERRQ(ierr);
  }
  // Set up mappings between local, global, dirichlet nodes.  Puts a global vector in c->force.
  ierr = StokesSetupDomain(c);CHKERRQ(ierr);
  { // Define matrices
    PetscInt n;
    ierr = VecDuplicate(c->force, X);CHKERRQ(ierr);
    ierr = VecGetSize(*X, &n);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))StokesMatMult);CHKERRQ(ierr);
  }
  {
    PetscInt m, n;
    PC pc;
    ierr = VecGetSize(c->velocityG, &n);CHKERRQ(ierr);
    ierr = VecGetSize(c->pressureG, &m);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, m, m, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatSchur);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatSchur, MATOP_MULT, (void(*)(void))StokesMatMultSchur);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatPV);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatPV, MATOP_MULT, (void(*)(void))StokesMatMultPV);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, m, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVP)CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVP, MATOP_MULT, (void(*)(void))StokesMatMultVP);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVV)CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVP, MATOP_MULT, (void(*)(void))StokesMatMultVV);CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(comm, d, n, n, 1+2*d, PETSC_NULL, &c->MatVVPC);CHKERRQ(ierr);
    ierr = KSPCreate(comm, &c->KSPSchur);CHKERRQ(ierr);
    ierr = KSPSetOperators(c->KSPSchur, c->MatSchur, c->MatSchur, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPGetPC(c->KSPSchur, &pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(c->KSPSchur, "schur_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(c->KSPSchur);CHKERRQ(ierr);
    ierr = KSPCreate(comm, &c->KSPVelocity);CHKERRQ(ierr);
    ierr = KSPSetOperators(c->KSPVelocity, c->MatVV, c->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(c->KSPVelocity, "vel_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(c->KSPVelocity);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesDestroy"
PetscErrorCode StokesDestroy (StokesCtx *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(c->KSPVelocity);CHKERRQ(ierr);
  ierr = KSPDestroy(c->KSPSchur);CHKERRQ(ierr);
  ierr = MatDestroy(c->MatSchur);CHKERRQ(ierr);
  ierr = MatDestroy(c->MatVelocity);CHKERRQ(ierr);
  ierr = MatDestroy(c->MatVelocityPC);CHKERRQ(ierr);
  for (int i=0; i < c->numDims; i++) {
    ierr = MatDestroy(c->Dfield[i]);CHKERRQ(ierr);
    ierr = MatDestroy(c->Dscalar[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(c->work, c->numWork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->workScalar, c->numWorkScalar);CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->workVel, c->numWorkVel);CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->gradu, c->numDims);CHKERRQ(ierr);
  ierr = VecDestroy(c->coord);CHKERRQ(ierr);
  ierr = VecDestroy(c->eta);CHKERRQ(ierr);
  ierr = VecDestroy(c->deta);CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet);CHKERRQ(ierr);
  ierr = VecDestroy(c->force);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLG);CHKERRQ(ierr);
  ierr = ISDestroy(c->isGL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLG);CHKERRQ(ierr);
  ierr = ISDestroy(c->isDL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLD);CHKERRQ(ierr);
  ierr = PetscFree(c->Dfield);CHKERRQ(ierr);
  ierr = PetscFree(c->dim);CHKERRQ(ierr);
  ierr = PetscFree2(c, c->options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesProcessOptions"
PetscErrorCode StokesProcessOptions(StokesCtx *ctx)
{
  MPI_Comm        comm = ctx->comm;
  StokesOptions  *opt = ctx->options;
  PetscTruth      flag;
  PetscInt        exact, boundary, rheology;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ctx->numDims = 10;
  ierr = PetscMalloc(ctx->numDims*sizeof(PetscInt), &ctx->dim);CHKERRQ(ierr);
  exact = 0; boundary = 0; rheology = 0;
  opt->debug = 0; opt->hardness = 1.0; opt->exponent = 1.0; opt->regularization = 1.0;
  ierr = PetscOptionsBegin(comm, "", "Stokes problem options", "");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dim", "list of dimension extent", "stokes.C", ctx->dim, &ctx->numDims, &flag);CHKERRQ(ierr);
  if (!flag) { ctx->numDims = 2; ctx->dim[0] = 8; ctx->dim[1] = 6; }
  ierr = PetscOptionsInt("-debug", "debugging level", "stokes.C", opt->debug, &opt->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-exact", "exact solution type", "stokes.C", exact, &exact, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-boundary", "boundary type", "stokes.C", boundary, &boundary, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rheology", "rheology type", "stokes.C", rheology, &rheology, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hardness", "power law hardness parameter", "stokes.C", opt->hardness, &opt->hardness, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-exponent", "power law exponent", "stokes.C", opt->exponent, &opt->exponent, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-regularization", "regularization parameter for viscosity", "stokes.C", opt->regularization, &opt->regularization, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscPrintf(comm, "Stokes problem");CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  dim = [", ctx->numDims);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    if (i > 0) { ierr = PetscPrintf(comm, ",");CHKERRQ(ierr); }
    ierr = PetscPrintf(comm, "%d", ctx->dim[i]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm, "]\n  hardness = %f    exponent = %8f    regularization = %8f\n", opt->hardness, opt->exponent, opt->regularization);CHKERRQ(ierr);

  switch (exact) {
    case 0:
      opt->exact    = StokesExactNull;
      opt->exactCtx = PETSC_NULL;
      break;
    case 1:
      opt->exact    = StokesExactCos;
      opt->exactCtx = PETSC_NULL;
      break;
    case 2:
      opt->exact    = StokesExactTest;
      opt->exactCtx = PETSC_NULL;
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Exact solution %d not implemented", exact);
  }
  switch (boundary) {
    case 0:
      opt->boundary    = StokesDirichlet;
      opt->boundaryCtx = (void *)opt->exact; // The dirichlet condition just evaluates the exact solution.
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Boundary type %d not implemented", exact);
  }
  switch (rheology) {
    case 0:
      opt->rheology    = StokesRheologyLinear;
      opt->rheologyCtx = PETSC_NULL;
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Rheology type %d not implemented", rheology);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMult"
PetscErrorCode StokesMatMult(Mat A, Vec xG, Vec yG)
{
  StokesCtx       *c;
  PetscInt         n, d, F;
  Vec              xL, *U, *V;
  PetscScalar    **u, **v, *eta, *x;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c);CHKERRQ(ierr);
  d = c->numDims; F = d+1; n = productInt(d, c->dim);
  xL = c->work[0]; U = &c->work[1]; V = &c->work[d+1];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(xL, c);CHKERRQ(ierr);

  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], xL, U[i]);CHKERRQ(ierr); }
  ierr = VecGetArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecGetArray(c->eta, &eta);CHKERRQ(ierr);
  // In `U' we have gradients [u_x v_x w_x p_x] [u_y v_y w_y p_y] [u_z v_z w_z p_z]
  for (int i=0; i < n; i++) { // each node
    for (int j=0; j < d; j++) { // each direction's derivative
      for (int f=0; f < d; f++) { // each velocity component
        u[j][i*F+f] = eta[i] * u[j][i*F+f]; // FIXME: nonlinear term
      }
      // Don't touch pressure component (important)
    }
  }
  ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->eta, &eta);CHKERRQ(ierr);

  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], U[i], V[i]);CHKERRQ(ierr); }
  // In `V' we have [(e u_x)_x (e v_x)_x (e w_x)_x p_xx] [(e u_y)_y (e v_y)_y (e w_y)_y p_yy] [...]
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
  ierr = VecGetArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
  for (int i=0; i < n; i++) { // each node
    for (int j=0; j < d; j++) {
      x[i*F+j] += u[j][i*F+d]; // pressure term
      x[i*F+d] += u[j][i*F+j]; // divergence of velocity
      for (int k=0; k < d; k++) {
        x[i*F+j] -= v[k][i*F+j]; // divergence of stress
      }
    }
  }
  ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);

  ierr = VecScatterBegin(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultSchur"
PetscErrorCode StokesMatMultSchur(Mat A, Vec xG, Vec yG)
{
  StokesCtx      *ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->MatVP, xG, ctx->velG);CHKERRQ(ierr);
  ierr = KSPSolve(ctx->KSPVel, ctx->velG, ctx->velG2);CHKERRQ(ierr);
  ierr = MatMult(ctx->MatPV, ctx->velG2, yG);CHKERRQ(ierr);
  ierr = VecScale(yG, -1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultPV"
PetscErrorCode StokesMatMultPV(Mat A, Vec xG, Vec yG)
{
  StokesCtx      *ctx;
  PetscReal     **u, *v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workVel[0]);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGV, xG, ctx->workVel[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGV, xG, ctx->workVel[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workScalar[2]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = VecStrideGather(ctx->workVel[0], i, ctx->workScalar[0], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatMult(ctx->Dscalar[i], ctx->workScalar[0], ctx->workScalar[1]);CHKERRQ(ierr);
    ierr = VecAXPY(ctx->workScalar[2], 1.0, ctx->workScalar[1]);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterPG, ctx->workScalar[2], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPG, ctx->workScalar[2], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultVP"
PetscErrorCode StokesMatMultVP(Mat A, Vec xG, Vec yG)
{
  StokesCtx      *ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workScalar[0]);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGP, xG, ctx->workScalar[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGP, xG, ctx->workScalar[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workVel[0]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = MatMult(ctx->Dscalar[i], ctx->workScalar[0], ctx->workScalar[1]);CHKERRQ(ierr);
    ierr = VecStrideScatter(ctx->workScalar[1], i, ctx->workVel[0], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterVG, ctx->workVel[0], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVG, ctx->workVel[0], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultVV"
PetscErrorCode StokesMatMultVV(Mat A, Vec xG, Vec yG)
{
  StokesCtx      *ctx;
  PetscInt        d, *dim, n;
  Vec             xL, *U, *V;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  d = ctx->numDims; dim = ctx->dim; n = productInt(d, dim);
  xL = c->workVel[0]; U = &c->workVel[1]; V = &c->workVel[d+1];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGV, xG, xL, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGV, xG, xL, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], xL, U[i]);CHKERRQ(ierr); }
  ierr = VecGetArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta);CHKERRQ(ierr);
  // In `U' we have gradients [u_x v_x w_x p_x] [u_y v_y w_y p_y] [u_z v_z w_z p_z]
  for (int i=0; i < n; i++) { // each node
    for (int j=0; j < d; j++) { // each direction's derivative
      for (int f=0; f < d; f++) { // each velocity component
        u[j][i*F+f] = eta[i] * u[j][i*F+f]; // FIXME: nonlinear term
      }
    }
  }
  ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->eta, &eta);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], U[i], V[i]);CHKERRQ(ierr); }
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
  ierr = VecGetArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
  for (int i=0; i < n; i++) { // each node
    for (int j=0; j < d; j++) { // velocity component
      for (int k=0; k < d; k++) { // direction of derivative
        x[i*d+j] -= v[k][i*d+j]; // divergence of stress
        // FIXME: There should be mixed terms here representing symmetrization
      }
    }
  }
  ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);

  ierr = VecScatterBegin(ctx->scatterVG, xL, yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVG, xL, yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesFunction"
PetscErrorCode StokesFunction(SNES snes, Vec xG, Vec rhs, void *ctx)
{
  StokesCtx *c = (StokesCtx *)ctx;
  PetscInt n, d, F;
  Vec xL, *V, *W;
  PetscReal **u, **v, **w, *eta, *deta, *x, *stretching;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  d = c->numDims; F = d+1; n = productInt(d, c->dim);
  xL = c->work[0]; V = &c->work[1]; W = &c->work[d+1];
  ierr = VecScatterBegin(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(xL, c);CHKERRQ(ierr);

  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], xL, c->gradu[i]);CHKERRQ(ierr); }
  //ierr = VecPrint2(c->gradu[0], "{u,v,w}_x", c);CHKERRQ(ierr);

  if (true) { // In `gradu' we have gradients [u_x v_x w_x p_x] [u_y v_y w_y p_y] [u_z v_z w_z p_z]
    PetscReal stretching[d*d];
    ierr = VecGetArrays(c->gradu, d, &u);CHKERRQ(ierr);
    ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
    ierr = VecGetArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecGetArray(c->deta, &deta);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      for (int j=0; j < d; j++) {
        for (int k=0; k < d; k++) {
          stretching[j*d+k] = 0.5 * (u[j][i*F+k] + u[k][i*F+j]);
        }
      }
      ierr = c->options->rheology(d, stretching, &eta[i], &deta[i], c->options->rheologyCtx);CHKERRQ(ierr);
      for (int j=0; j < d; j++) { // each direction's derivative
        for (int f=0; f < d; f++) { // each velocity component
          v[j][i*F+f] = eta[i] * u[j][i*F+f];
        }
        // Don't touch pressure component (important)
      }
    }
    ierr = VecRestoreArrays(c->gradu, d, &u);CHKERRQ(ierr);
    ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);
    ierr = VecRestoreArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecRestoreArray(c->deta, &deta);CHKERRQ(ierr);
  }

  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], V[i], W[i]);CHKERRQ(ierr); }

  { // In `W' we have [(e u_x)_x (e v_x)_x (e w_x)_x p_xx] [(e u_y)_y (e v_y)_y (e w_y)_y p_yy] [...]
    ierr = VecZeroEntries(xL);CHKERRQ(ierr);
    ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
    ierr = VecGetArrays(c->gradu, d, &u);CHKERRQ(ierr);
    ierr = VecGetArrays(W, d, &w);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      for (int j=0; j < d; j++) {
        x[i*F+j] += u[j][i*F+d]; // pressure term
        x[i*F+d] += u[j][i*F+j]; // divergence of velocity
        for (int k=0; k < d; k++) {
          x[i*F+j] -= w[k][i*F+j]; // divergence of stress
        }
      }
    }
    ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
    ierr = VecRestoreArrays(c->gradu, d, &u);CHKERRQ(ierr);
    ierr = VecRestoreArrays(W, d, &w);CHKERRQ(ierr);
  }

  //ierr = VecPrint2(xL, "discrete forcing", c);CHKERRQ(ierr);

  ierr = VecScatterBegin(c->scatterLG, xL, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, xL, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecAXPY(rhs, -1.0, c->force);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesJacobian"
PetscErrorCode StokesJacobian(SNES snes, Vec w, Mat *A, Mat *P, MatStructure *flag, void *void_ctx)
{
  StokesCtx     *ctx = (StokesCtx *)void_ctx;
  PetscInt       d = ctx->numDims, *dim = ctx->dim, F = d + 1;
  PetscReal     *x, *eta, *deta, **u0_;
  PetscInt      *ixL, n;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // The nonlinear term has already been fixed up by StokesFunction() so we just need to deal with the preconditioner here.
  ierr = PetscTypeCompare((PetscObject)*P, MATSHELL, &flg);CHKERRQ(ierr);
  if (flg) { printf("pc = shell\n"); PetscFunctionReturn(0); }
  ierr = VecGetSize(w, &n);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArrays(ctx->gradu, d, &u0_); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISGetIndices(ctx->isLG, &ixL); CHKERRQ(ierr);
  {
    PetscInt row[F], col[(2*d+1)*F];
    PetscScalar v[(2*d+1)*F];
    PetscInt c, c0, r;
    PetscScalar x0, xMM, xPP, xM, idxM, xP, idxP, idx, eM, deM, du0M, eP, deP, du0P;
    for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) { // loop over local dof
      const PetscInt i = it.i;
      for (int f=0; f < F; f++) { // Each equation, corresponds to a row in the matrix
        if (ixL[i*F+f] < 0 || n <= ixL[i*F+f]) continue; // Not a global dof
        c = 0; r = 0;
        c0 = c; row[r] = ixL[i*F+f]; col[c0] = row[r]; v[c0] = 0.0; r++; c++; // initialize diagonal term
        for (int j=0; j < d; j++) {
          const PetscInt iM = it.shift(j, -1);
          const PetscInt iP = it.shift(j,  1);
          if (iM < 0 || iP < 0) SETERRQ(1, "Local neighbor not on local grid.");
          x0 = x[i*d+j]; xMM = x[iM*d+j]; xPP = x[iP*d+j];
          xM = 0.5*(xMM+x0); idxM = 1.0/(x0-xMM); xP = 0.5*(x0+xPP); idxP = 1.0/(xPP-x0); idx = 1.0/(xP-xM);
          eM = 0.5*(eta[iM]+eta[i]); deM = 0.5*(deta[iM]+deta[i]); du0M = 0.5*(u0_[j][iM*F+f]+u0_[j][i*F+f]);
          eP = 0.5*(eta[iP]+eta[i]); deP = 0.5*(deta[iP]+deta[i]); du0P = 0.5*(u0_[j][iP*F+f]+u0_[j][i*F+f]);
          //printf("eta %f %f %f\n", eta[iM], eta[i], eta[iP]);
          deM = 0.0; du0M = 0.0; deP = 0.0; du0P = 0.0; // debugging
          if (f < d) { // velocity
            col[c] = ixL[iM*F+f]; v[c] = -idx * (idxM * eM - 0.5 * deM * du0M); c++;
            col[c] = ixL[iP*F+f]; v[c] = -idx * (idxP * eP + 0.5 * deP * du0P); c++;
            v[c0] += idx * (idxP * eP + idxM * eM - 0.5 * (deP * du0P - deM * du0M));
            if (f == j) { // add the pressure gradient
              col[c] = ixL[i *F+d]; v[c] = 0.5 * (idxM - idxP); c++;
              col[c] = ixL[iM*F+d]; v[c] = -0.5 * idxM; c++;
              col[c] = ixL[iP*F+d]; v[c] =  0.5 * idxP; c++;
            }
          } else { // pressure
            col[c] = ixL[i *F+j]; v[c] = 0.5 * (idxM - idxP); c++;
            col[c] = ixL[iM*F+j]; v[c] = -0.5 * idxM; c++;
            col[c] = ixL[iP*F+j]; v[c] =  0.5 * idxP; c++;
            v[c0] = 1.0e-8; // to avoid a zero on diagonal
          }
          //printf("i=%d f=%d j=%d row[..%d] = ",i,f,j,r); for (int i=0; i<r; i++) printf("%d ", row[i]); printf("\n");
          //printf("col[..%2d] = ", c); for (int i=0; i<c; i++) printf("%d ", col[i]); printf("\n");
        }
        ierr = MatSetValues(*P, r, row, c, col, v, INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecRestoreArrays(ctx->gradu, d, &u0_); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ctx->isLG, &ixL); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "StokesSetupDomain"
PetscErrorCode StokesSetupDomain(StokesCtx *c)
{
  MPI_Comm        comm = c->comm;
  PetscInt        d = c->numDims, *dim = c->dim, m = (d + 1) * productInt(d, dim);;
  IS              isG, isD;
  PetscInt       *ixL, *ixG, *ixD, *ind, l, g ,b;
  PetscReal      *uD, *x, *n, nn;
  BdyType         type;
  Vec             vL, vG, vD;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc5(m, PetscInt, &ixL, m, PetscInt, &ixG, m, PetscInt, &ixD, d, PetscInt, &ind, d, PetscReal, &n);CHKERRQ(ierr);
  ierr = VecGetArray(c->work[0], &uD);CHKERRQ(ierr); // Some workspace for boundary values
  ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector
  l = 0; g = 0; b = 0; // indices for local, global, and dirichlet boundary
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    for (int j=0; j < d; j++) {
      if (it.ind[j] == 0) {
        n[j] = -1.0;
      } else if (it.ind[j] == dim[j] - 1) {
        n[j] = 1.0;
      } else {
        n[j] = 0.0;
      }
    }
    nn = dotScalar(d, n, n);
    if (nn > 1e-5) { // We are on the boundary
      for (int j=0; j < d; j++) n[j] /= sqrt(nn); // normalize n
      ierr = c->options->boundary(d, &x[it.i*d], n, &type, &uD[b], c->options->boundaryCtx);CHKERRQ(ierr);
      if (type == DIRICHLET) {
        for (int k=0; k < d+1; k++) { // same mapping for each field
          ixL[l] = -1;
          ixD[b++] = l++;
        }
      } else { SETERRQ(1, "Neumann not implemented."); }
    } else { // Interior
      for (int k=0; k < d+1; k++) { // same mapping for each field
        ixL[l] = g;
        ixG[g++] = l++;
      }
    }
  }
  ierr = VecRestoreArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector

  ierr = VecCreate(comm, &vG);CHKERRQ(ierr);              // Prototype global vector
  ierr = VecSetSizes(vG, g, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vD);CHKERRQ(ierr);              // Special dirichlet vector
  ierr = VecSetSizes(vD, b, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vD);CHKERRQ(ierr);
  { // Fill dirichlet values into the special dirichlet vector.
    PetscScalar *v;
    ierr = VecGetArray(vD, &v);CHKERRQ(ierr);
    for (int i=0; i < b; i++) v[i] = uD[i];
    ierr = VecRestoreArray(vD, &v);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->work[0], &uD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "DOF distribution: %8d local     %8d global     %8d dirichlet\n", l, g, b);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, l, ixL, &c->isLG);CHKERRQ(ierr); // We need this to build the preconditioner
  ierr = ISCreateGeneral(comm, b, ixD, &c->isDL);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, g, ixG, &c->isGL);CHKERRQ(ierr);
  vL = c->work[0]; // A prototype local vector
  if (false) {
    PetscInt nL, nG, nD;
    ierr = VecGetSize(vL, &nL);CHKERRQ(ierr);
    ierr = VecGetSize(vG, &nG);CHKERRQ(ierr);
    ierr = VecGetSize(vD, &nD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Sizes: %8d local    %8d global    %8d dirichlet\n", nL, nG, nD);CHKERRQ(ierr);
    ierr = ISView(c->isDL, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isGL, &c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isGL, vG, PETSC_NULL, &c->scatterLG);CHKERRQ(ierr);
  ierr = PetscFree5(ixL, ixG, ixD, ind, n);CHKERRQ(ierr);
  c->force = vG;
  c->dirichlet = vD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesCreateExactSolution"
PetscErrorCode StokesCreateExactSolution(SNES snes, Vec U, Vec U2)
{
  StokesCtx      *c;
  PetscInt        d, *dim;
  PetscReal      *u, *v, *coord;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes, (void **)&c);CHKERRQ(ierr);
  d = c->numDims; dim = c->dim;
  ierr = VecGetArray(c->work[0], &u);CHKERRQ(ierr);
  ierr = VecGetArray(c->work[1], &v);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord);CHKERRQ(ierr);
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    const PetscInt i = it.i;
    ierr = c->options->exact(d, &coord[i*d], &u[i*(d+1)], &v[i*(d+1)], c->options->exactCtx);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->work[0], &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->work[1], &v);CHKERRQ(ierr);
  if (false) {
    ierr = VecPrint2(c->work[0], "exact solution", c);CHKERRQ(ierr);
    ierr = VecPrint2(c->work[1], "exact forcing", c);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(c->scatterLG, c->work[0], U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->work[0], U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->work[1], U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->work[1], U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLD, c->work[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLD, c->work[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecCopy(U2, c->force);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRemoveConstantPressure"
PetscErrorCode StokesRemoveConstantPressure(KSP ksp, StokesCtx *ctx, Vec *X, MatNullSpace *ns)
{
  MPI_Comm       comm;
  PetscInt       m;
  PetscReal     *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->work[0], &x);CHKERRQ(ierr);
  for (BlockIt it = BlockIt(ctx->numDims, ctx->dim); !it.done; it.next()) {
    const PetscInt i = it.i;
    for (int j=0; j < ctx->numDims; j++) {
      x[i*(ctx->numDims+1)+j] = 0.0;
    }
    x[i*(ctx->numDims+1)+ctx->numDims] = 1.0;
  }
  ierr = VecRestoreArray(ctx->work[0], &x);CHKERRQ(ierr);
  ierr = VecDuplicate(ctx->force, X);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterLG, ctx->work[0], *X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLG, ctx->work[0], *X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ksp, &comm);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, 1, X, ns);CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ksp, *ns);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPressureReduceOrder"
PetscErrorCode StokesPressureReduceOrder(Vec U, StokesCtx *c)
{
  PetscInt d = c->numDims, *dim = c->dim, F = d+1;
  PetscReal *u, *coord, *work, *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (d != 2) SETERRQ1(1, "Not implemented for dimension %d\n", d);
  //ierr = VecPrint2(U, "unfiltered", c);CHKERRQ(ierr);
  PetscInt m = dim[0], n = dim[1], mn = PetscMax(m,n);
  ierr = PetscMalloc2(4*mn, PetscReal, &work, mn, PetscReal, &x);CHKERRQ(ierr);
  ierr = VecGetArray(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord);CHKERRQ(ierr);
  for (int i=1; i < m; i++) {
    const PetscReal yM = coord[(i*n+0)*d+1];
    const PetscReal yP = coord[(i*n+n-1)*d+1];
    for (int j=0; j < n-2; j++) { // initialize Neville's algorithm
      x[j] = coord[(i*n+(j+1))*d+1]; // get `y` component
      work[j*4] = u[(i*n+(j+1))*F+d];
      work[j*4+1] = work[j*4];
    }
    ierr = polyInterp(n-2, x, work, yM, yP, &u[(i*n+0)*F+d], &u[(i*n+n-1)*F+d]);CHKERRQ(ierr);
  }
  for (int j=0; j < n; j++) { // This will set the corner values as well.
    const PetscReal xM = coord[(0*n+j)*d];
    const PetscReal xP = coord[((m-1)*n+j)*d];
    for (int i=0; i < m-2; i++) {
      x[i]  = coord[((i+1)*n+j)*d+0]; // get `x' component
      work[i*4] = u[((i+1)*n+j)*F+d];
      work[i*4+1] = work[i*4];
    }
    //for (int k=0; k < m-2; k++) { printf("%4f ", x[k]); } printf("\n");
    //for (int k=0; k < m-2; k++) { printf("%4f ", work[k*4]); } printf("\n");
    ierr = polyInterp(m-2, x, work, xM, xP, &u[(0*n+j)*F+d], &u[((m-1)*n+j)*F+d]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->coord, &coord);CHKERRQ(ierr);
  //ierr = VecPrint2(U, "filtered", c);CHKERRQ(ierr);
  ierr = PetscFree2(work, x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPCApply"
PetscErrorCode StokesPCApply(void *void_ctx, Vec x, Vec y)
{
  StokesCtx *ctx = (StokesCtx *)void_ctx;
  Vec xL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  xL = ctx->work[0];
  d = c->numDims; F = d+1; n = productInt(d, c->dim);
  xL = c->work[0]; U = &c->work[1]; V = &c->work[d+1];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // No need to remove constant pressure since it is the null space of the gradient
  // Assemble the right hand side, this requires a solve with the velocity matrix
  // Solve the Schur complement system

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRheologyLinear"
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta, void *ctx)
{

  PetscFunctionBegin; InFunction;
  *eta = 1.0;
  for (int i=0; i < d; i++) deta[i] = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesExactNull"
PetscErrorCode StokesExactNull(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{

  PetscFunctionBegin; InFunction;
  if (value) {
    for (int i=0; i < d+1; i++) value[i] = 0.0;
  }
  if (rhs) {
    for (int i=0; i < d+1; i++) rhs[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesExactCos"
PetscErrorCode StokesExactCos(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{
  const PetscReal eta = 1.0;
  PetscReal u, v, p;

  PetscFunctionBegin; InFunction;
  if (d != 2) SETERRQ2(1, "%s only implemented for dimension 2, %d given", __FUNCT__, d);
  u =  sin(0.5 * PETSC_PI * coord[0]) * cos(0.5 * PETSC_PI * coord[1]);
  v = -cos(0.5 * PETSC_PI * coord[0]) * sin(0.5 * PETSC_PI * coord[1]);
  p = 0.25 * (cos(PETSC_PI * coord[0]) + cos(PETSC_PI * coord[1])) + 10 * (coord[0] + coord[1]);
  if (value) {
    value[0] = u; value[1] = v; value[2] = p;
  }
  if (rhs) {
    rhs[0]   = 0.5 * PetscSqr(PETSC_PI) * eta * u - 0.25 * PETSC_PI * sin(PETSC_PI * coord[0]) + 10;
    rhs[1]   = 0.5 * PetscSqr(PETSC_PI) * eta * v - 0.25 * PETSC_PI * sin(PETSC_PI * coord[1]) + 10;
    rhs[2]   = 0.0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "StokesExactTest"
PetscErrorCode StokesExactTest(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{
  const PetscReal eta = 1.0;
  PetscReal u, v, p;

  PetscFunctionBegin; InFunction;
  if (d != 2) SETERRQ2(1, "%s only implemented for dimension 2, %d given", __FUNCT__, d);
  u =  sin(0.5 * PETSC_PI * coord[0]) * cos(0.5 * PETSC_PI * coord[1]);
  v = -cos(0.5 * PETSC_PI * coord[0]) * sin(0.5 * PETSC_PI * coord[1]);
  p = 0.0;
  if (value) {
    value[0] = u; value[1] = v; value[2] = p;
  }
  if (rhs) {
    rhs[0]   = 0.5 * PetscSqr(PETSC_PI) * eta * u;
    rhs[1]   = 0.5 * PetscSqr(PETSC_PI) * eta * v;
    rhs[2]   = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesDirichlet"
PetscErrorCode StokesDirichlet(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx)
{
  ExactSolution exact = (ExactSolution)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin; InFunction;
  *type = DIRICHLET;
  ierr = exact(d, coord, value, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPrint2"
PetscErrorCode VecPrint2(Vec v, const char *name, StokesCtx *ctx) {
  const char *field = "uvp";
  MPI_Comm comm;
  PetscScalar *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v, &comm); CHKERRQ(ierr);
  if (ctx->numDims == 2) {
    ierr = VecGetArray(v, &x); CHKERRQ(ierr);
    for (int f=0; f < 3; f++) {
      for (int j=0; j < ctx->dim[1]; j++) {
        ierr = PetscPrintf(comm, "%14s %c: ", name, field[f]);CHKERRQ(ierr);
        for (int i=ctx->dim[0]-1; i >= 0; i--) {
          ierr = PetscPrintf(comm, "%12.3e", x[(i*ctx->dim[1] + j)*3 + f]); CHKERRQ(ierr);
        }
        ierr = PetscPrintf(comm, "\n"); CHKERRQ(ierr);
      }
      if (f < 2) { ierr = PetscPrintf(comm, "-----------\n"); CHKERRQ(ierr); }
    }
    ierr = VecRestoreArray(v, &x); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "\n"); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
