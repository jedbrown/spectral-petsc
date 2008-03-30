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
typedef PetscErrorCode(*Rheology)(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx);

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
  PetscInt       numDims, numWorkP, numWorkV;
  PetscInt      *dim;
  KSP            KSPSchur, KSPVelocity;
  Mat            MatSchur, MatPV, MatVP, MatVV, MatVVPC; // Used by preconditioner
  Mat           *DP;            // derivative matrices for local pressure system
  Mat           *DV;            // derivative matrices for local velocity system
  Vec           *workP, *workV; // local work space
  Vec           *strain;        // each is vector valued, numDims
  Vec            eta, deta;     // effective viscosity and it's derivative w.r.t second invariant, scalar valued
  Vec            coord;         // vector valued, numDims
  Vec            dirichlet;     // special dirichlet format (velocity only, pressure does not have boundary conditions)
  Vec            force;         // Global body force vector, velocity only
  Vec            pG0, pG1, vG0, vG1; // global pressure and velocity vectors
  IS             isLV, isVL, isLD;     // velocity local <-> global, dirichlet
  VecScatter     scatterLP, scatterPL; // pressure local <-> pressure global
  VecScatter     scatterLV, scatterVL; // velocity local <-> global
  VecScatter     scatterLD, scatterDL; // velocity local <-> special dirichlet
  VecScatter     scatterPG, scatterGP; // global pressure <-> full global
  VecScatter     scatterVG, scatterGV; // global velocity <-> full global
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
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx);
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
  PetscInt        d, *dim, m;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(1, StokesCtx, ctx, 1, StokesOptions, &opt);CHKERRQ(ierr);
  c = *ctx; c->options = opt; c->comm = comm;
  ierr = StokesProcessOptions(c);CHKERRQ(ierr);
  d = c->numDims; dim = c->dim; m = productInt(d, dim);
  ierr = PetscMalloc2(d, Mat, &c->DP, d, Mat, &c->DV);CHKERRQ(ierr);
  c->numWorkP = 2*d+1; c->numWorkV = 2*d+1;
  ierr = VecCreate(comm, &c->eta);CHKERRQ(ierr); // Prototype for scalar valued local
  ierr = VecSetSizes(c->eta, m, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c->eta);CHKERRQ(ierr);
  ierr = VecCreate(comm, &c->coord);CHKERRQ(ierr); // prototype for vector valued local
  ierr = VecSetSizes(c->coord, m*d, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->coord, d);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c->coord);CHKERRQ(ierr);
  ierr = VecDuplicate(c->eta, c->deta);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->eta, c->numWorkP, &c->workP);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->coord, c->numWorkV, &c->workV);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->coord, d, &c->strain);CHKERRQ(ierr);
  { // Create differentiation matrices
    PetscInt cheb_dim[d+1];
    for (int i=0; i < d; i++) { cheb_dim[i] = dim[i]; }
    cheb_dim[d] = d;
    for (int i=0; i < d; i++) {
      ierr = MatCreateCheb(comm, d, i, dim, fftw_flag, c->workP[0], c->workP[1], &c->DP[i]);CHKERRQ(ierr);
      ierr = MatCreateCheb(comm, d+1, i, cheb_dim, fftw_flag, c->workV[0], c->workV[1], &c->DV[i]);CHKERRQ(ierr);
    }
  }
  { // Set up coordinate mapping
    PetscReal *x;
    ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);
    for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
      for (int j=0; j < d; j++) {
        x[it.i*d+j] = cos(it.ind[j] * PETSC_PI / (dim[j] - 1));
      }
    }
    ierr = VecRestoreArray(c->coord, &x);CHKERRQ(ierr);
  }
  // Set up mappings between local, global, dirichlet nodes.  Returns a global system vector.
  ierr = StokesSetupDomain(c, X);CHKERRQ(ierr);
  { // Define global system matrix
    PetscInt n;
    ierr = VecGetSize(*X, &n);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))StokesMatMult);CHKERRQ(ierr);
  }
  { // Inner matrices
    PetscInt m, n;
    PC pc;
    ierr = VecGetSize(c->pG0, &m);CHKERRQ(ierr);
    ierr = VecGetSize(c->vG0, &n);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, m, m, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatSchur);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatSchur, MATOP_MULT, (void(*)(void))StokesMatMultSchur);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatPV);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatPV, MATOP_MULT, (void(*)(void))StokesMatMultPV);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, m, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVP)CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVP, MATOP_MULT, (void(*)(void))StokesMatMultVP);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVV)CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVV, MATOP_MULT, (void(*)(void))StokesMatMultVV);CHKERRQ(ierr);
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
  ierr = MatDestroy(c->MatPV);CHKERRQ(ierr);                  ierr = MatDestroy(c->MatVP);CHKERRQ(ierr);
  ierr = MatDestroy(c->MatVV);CHKERRQ(ierr);                  ierr = MatDestroy(c->MatVVPC);CHKERRQ(ierr);
  for (int i=0; i < c->numDims; i++) {
    ierr = MatDestroy(c->DP[i]);CHKERRQ(ierr);
    ierr = MatDestroy(c->DV[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(c->workP, c->numWorkP);CHKERRQ(ierr); ierr = VecDestroyVecs(c->workV, c->numWorkV);CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->strain, c->numDims);CHKERRQ(ierr);
  ierr = VecDestroy(c->eta);CHKERRQ(ierr);                    ierr = VecDestroy(c->deta);CHKERRQ(ierr);
  ierr = VecDestroy(c->coord);CHKERRQ(ierr);                  ierr = VecDestroy(c->dirichlet);CHKERRQ(ierr);
  ierr = VecDestroy(c->force);CHKERRQ(ierr);
  ierr = VecDestroy(c->pG0);CHKERRQ(ierr);                    ierr = VecDestroy(c->pG1);CHKERRQ(ierr);
  ierr = VecDestroy(c->vG0);CHKERRQ(ierr);                    ierr = VecDestroy(c->vG1);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLV);CHKERRQ(ierr);                    ierr = ISDestroy(c->isVL);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLP);CHKERRQ(ierr);       ierr = VecScatterDestroy(c->scatterPL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLV);CHKERRQ(ierr);       ierr = VecScatterDestroy(c->scatterVL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLD);CHKERRQ(ierr);       ierr = VecScatterDestroy(c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterPG);CHKERRQ(ierr);       ierr = VecScatterDestroy(c->scatterGP);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterVG);CHKERRQ(ierr);       ierr = VecScatterDestroy(c->scatterGV);CHKERRQ(ierr);
  ierr = PetscFree2(c->DP, c->DV);CHKERRQ(ierr);
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
  ierr = MatMult(ctx->MatVP, xG, ctx->vG0);CHKERRQ(ierr);
  ierr = KSPSolve(ctx->KSPVel, ctx->vG0, ctx->vG1);CHKERRQ(ierr);
  ierr = MatMult(ctx->MatPV, ctx->vG1, yG);CHKERRQ(ierr);
  ierr = VecScale(yG, -1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultPV"
PetscErrorCode StokesMatMultPV(Mat A, Vec xG, Vec yG) // divergence of velocity
{
  StokesCtx      *ctx;
  PetscReal     **u, *v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workV[0]);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, xG, ctx->workV[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, xG, ctx->workV[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workP[2]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = VecStrideGather(ctx->workV[0], i, ctx->workP[0], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    ierr = VecAXPY(ctx->workP[2], 1.0, ctx->workP[1]);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLP, ctx->workP[2], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLP, ctx->workP[2], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultVP"
PetscErrorCode StokesMatMultVP(Mat A, Vec xG, Vec yG) // gradient of pressure
{
  StokesCtx      *ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workP[0]);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterPL, xG, ctx->workP[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPL, xG, ctx->workP[0], SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(ctx->workP[0], c);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workV[0]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    ierr = VecStrideScatter(ctx->workP[1], i, ctx->workV[0], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLV, ctx->workV[0], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLV, ctx->workV[0], yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMatMultVV"
PetscErrorCode StokesMatMultVV(Mat A, Vec xG, Vec yG) // Jacobian of viscous term
{
  StokesCtx      *ctx;
  PetscInt        d, *dim, n;
  Vec             xL, yL, *V;
  PetscReal      *eta, *deta, **v, **Strain;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  d = ctx->numDims; dim = ctx->dim; n = productInt(d, dim);
  xL = c->workV[0]; yL = c->workV[1]; V = &c->workV[2];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGV, xG, xL, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGV, xG, xL, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(c->DV[i], xL, V[i]);CHKERRQ(ierr); }
  ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->strain, &Strain);CHKERRQ(ierr);
  // In `U' we have gradients [u_x v_x w_x p_x] [u_y v_y w_y p_y] [u_z v_z w_z p_z]
  for (int i=0; i < n; i++) { // each node
    PetscReal strain[d][d], z=0.0;
    for (int j=0; j < d; j++) { // each direction's derivative
      for (int k=0; k < d; k++) { // each velocity component
        strain[j][k] = 0.5 * (v[j][i*d+k] + v[k][i*d+j]);
        z += strain[j][k] * Strain[j][i*d+k];
      }
    }
    for (int j=0; j < d; j++) { // each direction's derivative
      for (int k=0; k < d; k++) { // each velocity component
        v[j][i*d+k] = eta[i] * strain[j][k] + deta[i] * Strain[j][i*d+k] * z;
      }
    }
  }
  ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->eta, &eta);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->deta, &deta);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->strain, &Strain);CHKERRQ(ierr);
  ierr = VecZeroEntries(yL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) {
    ierr = MatMult(c->DV[i], V[i], xL);CHKERRQ(ierr);
    ierr = VecAXPY(yL, -1.0, xL);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLV, yL, yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLV, yL, yG, SCATTER_FORWARD, INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesFunction"
PetscErrorCode StokesFunction(SNES snes, Vec xG, Vec rhs, void *ctx)
{
  StokesCtx       *c = (StokesCtx *)ctx;
  PetscInt         n, d, F;
  Vec              xL, yL, *V;
  PetscReal      **v, **strain, *eta, *deta;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  d = c->numDims; F = d+1; n = productInt(d, c->dim);
  xL = c->workV[0]; yL = c->workV[1]; V = &c->workV[2];
  ierr = VecScatterBegin(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVL, c->vG0, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVL, c->vG0, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  for (int i=0; i < d; i++) { ierr = MatMult(c->DV[i], xL, c->strain[i]);CHKERRQ(ierr); }

  if (true) { // Symmetrize strain, compute nonlinear contributions
    PetscReal s[d][d], gamma=0.0;
    ierr = VecGetArrays(c->strain, d, &strain);CHKERRQ(ierr);
    ierr = VecGetArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecGetArray(c->deta, &deta);CHKERRQ(ierr);
    ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      for (int j=0; j < d; j++) { // each direction
        for (int k=0; k < d; k++) { // each component of velocity
          s[j][k] = 0.5 * (strain[j][i*d+k] + strain[k][i*d+j]);
          gamma += 0.5 * PetscSqr(s[j][k]);
        }
      }
      ierr = c->options->rheology(d, gamma, &eta[i], &deta[i], c->options->rheologyCtx);CHKERRQ(ierr);
      for (int j=0; j < d; j++) { // each direction's derivative
        for (int k=0; k < d; k++) { // each velocity component
          strain[j][i*d+f] = s[j][k]; // store the strain
          v[j][i*d+k] = eta[i] * s[j][k]; // part of function evaluation
        }
      }
    }
    ierr = VecRestoreArrays(c->strain, d, &strain);CHKERRQ(ierr);
    ierr = VecRestoreArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecRestoreArray(c->deta, &deta);CHKERRQ(ierr);
    ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(yL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) {
    ierr = MatMult(c->DV[i], V[i], xL);CHKERRQ(ierr);
    ierr = VecAXPY(yL, -1.0, xL);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(c->scatterLV, yL, c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLV, yL, c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // This function is now completely done with local vecs, so it is safe to apply these matrices.
  ierr = MatMult(c->MatPV, c->vG0, c->pG1);CHKERRQ(ierr); // divergence of velocity
  ierr = MatMult(c->MatVP, c->pG1, c->vG0);CHKERRQ(ierr); // gradient of pressure
  ierr = VecAXPY(c->vG1, 1.0, c->vG0);CHKERRQ(ierr);
  ierr = VecAXPY(rhs, -1.0, c->force);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->vP1, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->vP1, rhs, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
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
  PetscInt        d = c->numDims, *dim = c->dim, n = productInt(d, dim), m = d*n, N=m+n;
  IS              isG, isD;
  PetscInt       *ixL, *ixG, *ixD, *ixLp, *ixGp, *ixDp, *ixLv, *ixGv, *ixDv;
  PetscInt        l, g ,b, lp, gp, bp, lv, gv, bv;
  PetscReal      *uD, *pD, *vD, *u, *v, *p, *x, *n, nn;
  BdyType         type;
  Vec             uL, uG, uD, pL, pG, pD, vL, vG, vD;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc4(N, PetscInt, &ixL, N, PetscInt, &ixG, N, PetscInt, &ixD, d, PetscReal, &n);CHKERRQ(ierr);
  ierr = PetscMalloc6(m, PetscInt, &ixLp, m, PetscInt, &ixGp, m, PetscInt, &ixDp, n, PetscInt &ixLv, n, PetscInt, &ixGv, n, PetscInt, &ixDv);CHKERRQ(ierr);
  ierr = VecGetArray(c->work[0], &uD);CHKERRQ(ierr); // Some workspace for boundary values
  ierr = VecGetArray(c->workScalar[0], &pD);CHKERRQ(ierr);
  ierr = VecGetArray(c->workVel[0], &vD);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector
  l = 0; g = 0; b = 0; // indices for local, global, and dirichlet boundary
  lp = 0; gp = 0; bp = 0; lv = 0; gv = 0; bv = 0; // for pressure and velocity subsystems
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    for (int j=0; j < d; j++) { // Compute normal vector
      if      (it.ind[j] == 0)          { n[j] = -1.0; }
      else if (it.ind[j] == dim[j] - 1) { n[j] = 1.0;  }
      else                              { n[j] = 0.0;  }
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
        pD[bp] = uD[b+d];
        ixLp[lp] = -1; ixDp[bp++] = lp++; // mapping for just pressure
        for (int k=0; k < d; k++) { vD[bv] = uD[b+k]; ixLv[lv] = -1; ixDv[bv++] = lv++; } // mapping for just velocity
      } else { SETERRQ(1, "Neumann not implemented."); }
    } else { // Interior
      for (int k=0; k < d+1; k++) { // same mapping for each field
        ixL[l] = g;
        ixG[g++] = l++;
      }
      ixLp[lp] = gp; ixGp[gp++] = lp++; // mapping for just pressure
      for (int k=0; k < d; k++) { ixLv[lv] = gv; ixGv[gv++] = lv++; } // mapping for just velocity
    }
  }
  ierr = VecRestoreArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector
  ierr = ISCreateGeneral(comm, l, ixL, &c->isLG);CHKERRQ(ierr); // We need this to build the preconditioner
  ierr = ISCreateGeneral(comm, g, ixG, &c->isGL);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, b, ixD, &c->isDL);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, lp, ixLp, &c->isLGp);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, gp, ixGp, &c->isGLp);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, bp, ixDp, &c->isDLp);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, lv, ixLv, &c->isLGv);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, gv, ixGv, &c->isGLv);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, bv, ixDv, &c->isDLv);CHKERRQ(ierr);
  { // Create global and dirichlet vectors for full system
    ierr = VecCreate(comm, &uG);CHKERRQ(ierr);              // Prototype global vector
    ierr = VecSetSizes(uG, g, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(uG);CHKERRQ(ierr);
    ierr = VecCreate(comm, &uD);CHKERRQ(ierr);              // Special dirichlet vector
    ierr = VecSetSizes(uD, b, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(uD);CHKERRQ(ierr);
    // Fill dirichlet values into the special dirichlet vector.
    ierr = VecGetArray(uD, &u);CHKERRQ(ierr);
    for (int i=0; i < b; i++) u[i] = uD[i];
    ierr = VecRestoreArray(uD, &u);CHKERRQ(ierr);
  }
  { // Create global and dirichlet vectors for pressure system
    ierr = VecCreate(comm, &pG);CHKERRQ(ierr);              // Prototype global vector
    ierr = VecSetSizes(pG, gp, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(pG);CHKERRQ(ierr);
    ierr = VecCreate(comm, &pD);CHKERRQ(ierr);              // Special dirichlet vector
    ierr = VecSetSizes(pD, bp, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(pD);CHKERRQ(ierr);
    ierr = VecGetArray(pD, &p);CHKERRQ(ierr);
    for (int i=0; i < b; i++) p[i] = pD[i];
    ierr = VecRestoreArray(pD, &p);CHKERRQ(ierr);
  }
  { // Create global and dirichlet vectors for pressure system
    ierr = VecCreate(comm, &vG);CHKERRQ(ierr);              // Prototype global vector
    ierr = VecSetSizes(vG, gv, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vG);CHKERRQ(ierr);
    ierr = VecCreate(comm, &vD);CHKERRQ(ierr);              // Special dirichlet vector
    ierr = VecSetSizes(vD, bv, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vD);CHKERRQ(ierr);
    ierr = VecGetArray(vD, &v);CHKERRQ(ierr);
    for (int i=0; i < b; i++) v[i] = vD[i];
    ierr = VecRestoreArray(vD, &v);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->work[0], &uD);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workScalar[0], &pD);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workVec[0], &vD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "    full DOF distribution: %8d local     %8d global     %8d dirichlet\n", l, g, b);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "pressure DOF distribution: %8d local     %8d global     %8d dirichlet\n", lp, gp, bp);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "velocity DOF distribution: %8d local     %8d global     %8d dirichlet\n", lv, gv, bv);CHKERRQ(ierr);
  uL = c->work[0]; pL = c->workScalar[0]; vL = c->workVel[0]; // prototype local vectors
  // Scatters for full system
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isGL, &c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isGL, vG, PETSC_NULL, &c->scatterLG);CHKERRQ(ierr);
  // Scatters for pressure system
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isGL, &c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isGL, vG, PETSC_NULL, &c->scatterLG);CHKERRQ(ierr);
  // Scatters for velocity system
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isGL, &c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isGL, vG, PETSC_NULL, &c->scatterLG);CHKERRQ(ierr);

  ierr = PetscFree4(ixL, ixG, ixD, n);CHKERRQ(ierr);
  ierr = PetscFree6(ixLp, ixGp, ixDp, 
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
  ierr = VecNormalize(*X);CHKERRQ(ierr);
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
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx)
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
