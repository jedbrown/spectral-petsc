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
  MatNullSpace   NullSpaceSchur;
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
  IS             isLP, isPL, isLV, isVL, isLD, isDL, isPG, isGP, isVG, isGV; // perhaps useful for building preconditioner
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
PetscErrorCode StokesDivergence(StokesCtx *ctx, PetscTruth withDirichlet, Vec xG, Vec yG);
PetscErrorCode StokesFunction(SNES, Vec, Vec, void *);
PetscErrorCode StokesJacobian(SNES, Vec, Mat *, Mat *, MatStructure *, void *);
PetscErrorCode StokesSetupDomain(StokesCtx *, Vec *);
PetscErrorCode StokesCreateExactSolution(SNES, Vec u, Vec u2);
PetscErrorCode StokesCheckResidual(SNES snes, Vec u, Vec x);
PetscErrorCode StokesRemoveConstantPressure(KSP, StokesCtx *, Vec *, MatNullSpace *);
PetscErrorCode StokesPressureReduceOrder(Vec u, StokesCtx *c);
PetscErrorCode StokesPCApply(void *, Vec, Vec);
PetscErrorCode StokesSchurMatMult(Mat, Vec, Vec);
PetscErrorCode StokesVelocityMatMult(Mat, Vec, Vec);

//PetscErrorCode StokesRheologyPowerLaw(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta);
PetscErrorCode VecPrint2(Vec v, PetscInt m, PetscInt n, PetscInt F, const char *name, const char *component);
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
  {
    PetscInt m, n;
    ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
    ierr = MatCreate(comm, &P);CHKERRQ(ierr);
    ierr = MatSetSizes(P, m, n, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  }
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
  ierr = VecPrint2(u2, ctx->dim[0]-2, ctx->dim[1]-2, 3, "exact gforce", "uvp");CHKERRQ(ierr);
  ierr = StokesFunction(snes, u, r, ctx);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, &unorm);CHKERRQ(ierr);
  ierr = VecNorm(u2, NORM_INFINITY, &u2norm);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_INFINITY, &rnorm);CHKERRQ(ierr);
  {
    ierr = PetscPrintf(comm, "Norm of solution %9.3e  norm of forcing %9.3e  norm of residual %9.3e\n", unorm, u2norm, rnorm);CHKERRQ(ierr);
    if (ctx->options->debug > 0) {
      ierr = VecPrint2(u,  ctx->dim[0]-2, ctx->dim[1]-2, 3, "exact global", "uvp");CHKERRQ(ierr);
      //ierr = VecPrint2(u2, ctx->dim[0]-2, ctx->dim[1]-2, 3, "exact gforce", "uvp");CHKERRQ(ierr);
      ierr = VecPrint2(r,  ctx->dim[0]-2, ctx->dim[1]-2, 3, "exact residual", "uvp");CHKERRQ(ierr);
      //ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);printf("\n");
      //ierr = VecView(u2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);printf("\n");
      //ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);printf("\n");
    }
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
  ierr = VecDuplicate(c->eta, &c->deta);CHKERRQ(ierr);
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
    ierr = MatCreateShell(comm, n, m, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVP);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVP, MATOP_MULT, (void(*)(void))StokesMatMultVP);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, &c->MatVV);CHKERRQ(ierr);
    ierr = MatShellSetOperation(c->MatVV, MATOP_MULT, (void(*)(void))StokesMatMultVV);CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(comm, n, n, 1+2*d, PETSC_NULL, &c->MatVVPC);CHKERRQ(ierr);
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
  ierr = MatNullSpaceDestroy(c->NullSpaceSchur);CHKERRQ(ierr);
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
  ierr = ISDestroy(c->isLP);CHKERRQ(ierr);                    ierr = ISDestroy(c->isPL);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLV);CHKERRQ(ierr);                    ierr = ISDestroy(c->isVL);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLD);CHKERRQ(ierr);                    ierr = ISDestroy(c->isDL);CHKERRQ(ierr);
  ierr = ISDestroy(c->isPG);CHKERRQ(ierr);                    ierr = ISDestroy(c->isGP);CHKERRQ(ierr);
  ierr = ISDestroy(c->isVG);CHKERRQ(ierr);                    ierr = ISDestroy(c->isGV);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(c->MatVV, c->vG0, c->vG1);CHKERRQ(ierr);
  ierr = MatMult(c->MatPV, c->vG0, c->pG1);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(c->MatVP, c->pG0, c->vG0);CHKERRQ(ierr);
  ierr = VecAXPY(c->vG1, 1.0, c->vG0);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->pG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = KSPSolve(ctx->KSPVelocity, ctx->vG0, ctx->vG1);CHKERRQ(ierr);
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
  ierr = StokesDivergence(ctx, PETSC_FALSE, xG, yG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesDivergence"
PetscErrorCode StokesDivergence(StokesCtx *ctx, PetscTruth withDirichlet, Vec xG, Vec yG) // divergence of velocity
{
  PetscReal     **u, *v;
  PetscErrorCode  ierr;

  ierr = VecZeroEntries(ctx->workV[0]);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, xG, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, xG, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  if (withDirichlet) {
    ierr = VecScatterBegin(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  }
  //ierr = VecView(ctx->dirichlet, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workP[2]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = VecStrideGather(ctx->workV[0], i, ctx->workP[0], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    //ierr = VecPrint2(ctx->workP[0],  ctx->dim[0], ctx->dim[1], 1, "uv component", "?");CHKERRQ(ierr);
    //ierr = VecPrint2(ctx->workP[1],  ctx->dim[0], ctx->dim[1], 1, "uv gradient", "?");CHKERRQ(ierr);
    ierr = VecAXPY(ctx->workP[2], 1.0, ctx->workP[1]);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLP, ctx->workP[2], yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLP, ctx->workP[2], yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(ctx->scatterPL, xG, ctx->workP[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPL, xG, ctx->workP[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(ctx->workP[0], ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workV[0]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    ierr = VecStrideScatter(ctx->workP[1], i, ctx->workV[0], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLV, ctx->workV[0], yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLV, ctx->workV[0], yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
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
  xL = ctx->workV[0]; yL = ctx->workV[1]; V = &ctx->workV[2];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(ctx->DV[i], xL, V[i]);CHKERRQ(ierr); }
  ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta);CHKERRQ(ierr);
  ierr = VecGetArrays(ctx->strain, d, &Strain);CHKERRQ(ierr);
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
  ierr = VecRestoreArrays(ctx->strain, d, &Strain);CHKERRQ(ierr);
  ierr = VecZeroEntries(yL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) {
    ierr = MatMult(ctx->DV[i], V[i], xL);CHKERRQ(ierr);
    ierr = VecAXPY(yL, -1.0, xL);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(ctx->scatterLV, yL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLV, yL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesFunction"
PetscErrorCode StokesFunction(SNES snes, Vec xG, Vec yG, void *ctx)
{
  StokesCtx       *c = (StokesCtx *)ctx;
  PetscInt         n, d, F;
  Vec              xL, yL, *V;
  PetscReal      **v, **strain, *eta, *deta;
  PetscErrorCode   ierr;

  PetscFunctionBegin; InFunction;
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
          v[j][i*d+k] = eta[i] * s[j][k]; // part of function evaluation
          strain[j][i*d+k] = s[j][k]; // store the strain rate
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
  //ierr = VecPrint2(yL,  c->dim[0], c->dim[1], 2, "viscous", "uv");CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLV, yL, c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLV, yL, c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // This function is now completely done with local vecs, so it is safe to apply these matrices.
  ierr = StokesDivergence(c, PETSC_TRUE, c->vG0, c->pG1);CHKERRQ(ierr); // divergence of velocity
  ierr = MatMult(c->MatVP, c->pG0, c->vG0);CHKERRQ(ierr); // gradient of pressure
  //ierr = VecPrint2(c->pG0,  c->dim[0]-2, c->dim[1]-2, 1, "pressure", "p");CHKERRQ(ierr);
  //ierr = VecPrint2(c->vG0,  c->dim[0]-2, c->dim[1]-2, 2, "gradient", "uv");CHKERRQ(ierr);
  ierr = VecAXPY(c->vG1, 1.0, c->vG0);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->pG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG1, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  //ierr = VecPrint2(c->force,  c->dim[0]-2, c->dim[1]-2, 3, "stored f", "uvp");CHKERRQ(ierr);
  ierr = VecAXPY(yG, -1.0, c->force);CHKERRQ(ierr);
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
#if 0
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
#endif
  *flag = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesSetupDomain"
PetscErrorCode StokesSetupDomain(StokesCtx *c, Vec *global)
{
  MPI_Comm        comm = c->comm;
  PetscInt        d = c->numDims, *dim = c->dim, m = productInt(d, dim), n = d*m, N=n+m;
  IS              isG, isD;
  PetscInt       *ixLP, *ixPL, *ixLV, *ixVL, *ixLD, *ixDL, *ixPG, *ixGP, *ixVG, *ixGV;
  PetscInt        lp, lv, gp, gv, dv, g;
  PetscReal      *v, *w, *x;
  BdyType         type;
  Vec             uG, pL, pG, vL, vG, vD;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc6(m, PetscInt, &ixLP, m, PetscInt, &ixPL, n, PetscInt, &ixLV, n, PetscInt, &ixVL, n, PetscInt, &ixLD, n, PetscInt, &ixDL);CHKERRQ(ierr);
  ierr = PetscMalloc4(m, PetscInt, &ixPG, N, PetscInt, &ixGP, n, PetscInt, &ixVG, N, PetscInt, &ixGV);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector
  ierr = VecGetArray(c->workV[0], &v);CHKERRQ(ierr); // Some workspace for boundary values
  lp=lv=gp=gv=dv=g=0;
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    PetscReal normal[d], normal2;
    for (int j=0; j < d; j++) { // Compute normal vector
      if      (it.ind[j] == 0)          { normal[j] = -1.0; }
      else if (it.ind[j] == dim[j] - 1) { normal[j] = 1.0;  }
      else                              { normal[j] = 0.0;  }
    }
    normal2 = dotScalar(d, normal, normal);
    if (normal2 > 1e-5) { // We are on the boundary
      for (int j=0; j < d; j++) normal[j] /= sqrt(normal2); // normalize n
      ierr = c->options->boundary(d, &x[it.i*d], normal, &type, &v[dv], c->options->boundaryCtx);CHKERRQ(ierr);
      if (type == DIRICHLET) {
        for (int k=0; k < d; k++) { // velocity in the local system comes from dirichlet values
          ixLV[lv] = -1;                            // local dof is not in the global system
          ixDL[dv] = lv; ixLD[lv] = dv; lv++; dv++; // 2-way mapping from local velocity to dirichlet
        }
      } else { SETERRQ(1, "Neumann not implemented."); }
      ixLP[lp++] = -1; // local pressure is not represented in global pressure
    } else { // Interior
      for (int k=0; k < d; k++) {
        ixLV[lv] = gv; ixVL[gv] = lv; // two way map from local to global velocity
        ixVG[gv] = g;  ixGV[g ] = gv; // two way map from global velocity to full global
        ixGP[g] = -1;
        ixLD[lv] = -1;
        lv++; gv++; g++;
      }
      ixLP[lp] = gp; ixPL[gp] = lp; // two way map from local pressure to global pressure
      ixPG[gp] = g;  ixGP[g ] = gp; // two way map from global pressure to full global
      ixGV[g] = -1;
      lp++; gp++; g++;
    }
  }
  ierr = VecRestoreArray(c->coord, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workV[0], &v);CHKERRQ(ierr);
  ierr = VecCreate(comm, &uG);CHKERRQ(ierr);   ierr = VecSetSizes(uG,  g, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(uG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &pG);CHKERRQ(ierr);   ierr = VecSetSizes(pG, gp, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(pG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vG);CHKERRQ(ierr);   ierr = VecSetSizes(vG, gv, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(vG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vD);CHKERRQ(ierr);   ierr = VecSetSizes(vD, dv, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(vD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "DOF distribution: %d global   %d/%d pressure    %d/%d velocity    %d dirichlet\n", g, gp, lp, gv, lv, dv);CHKERRQ(ierr);
  { // These index sets are needed to create the scatters, but may be needed when forming preconditioners, so we store them in StokesCtx.
    ierr = ISCreateGeneral(comm, lp, ixLP, &c->isLP);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, gp, ixPL, &c->isPL);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, lv, ixLV, &c->isLV);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, gv, ixVL, &c->isVL);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, lv, ixLD, &c->isLD);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, dv, ixDL, &c->isDL);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, gp, ixPG, &c->isPG);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, g , ixGP, &c->isGP);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, gv, ixVG, &c->isVG);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, g , ixGV, &c->isGV);CHKERRQ(ierr);
  }
  pL = c->workP[0]; vL = c->workV[1]; // prototype local vectors
  ierr = VecScatterCreate(pL, c->isPL, pG, PETSC_NULL, &c->scatterLP);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isVL, vG, PETSC_NULL, &c->scatterLV);CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD);CHKERRQ(ierr);
  ierr = VecScatterCreate(pG, PETSC_NULL, pL, c->isPL, &c->scatterPL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isVL, &c->scatterVL);CHKERRQ(ierr);
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL);CHKERRQ(ierr);
  if (false) {
    ierr = VecSet(pL, 1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(c->scatterLP, pL, pG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(c->scatterLP, pL, pG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecView(pG, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecSet(pG, 2.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(c->scatterPL, pG, pL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(c->scatterPL, pG, pL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecView(pL, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(pG, PETSC_NULL, uG, c->isPG, &c->scatterPG);CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, uG, c->isVG, &c->scatterVG);CHKERRQ(ierr);
  ierr = VecScatterCreate(uG, c->isPG, pG, PETSC_NULL, &c->scatterGP);CHKERRQ(ierr);
  ierr = VecScatterCreate(uG, c->isVG, vG, PETSC_NULL, &c->scatterGV);CHKERRQ(ierr);
  ierr = PetscFree6(ixLP, ixPL, ixLV, ixVL, ixLD, ixDL);CHKERRQ(ierr);
  ierr = PetscFree4(ixPG, ixGP, ixVG, ixGV);CHKERRQ(ierr);
  c->pG0 = pG; ierr = VecDuplicate(pG, &c->pG1);CHKERRQ(ierr);
  c->vG0 = vG; ierr = VecDuplicate(vG, &c->vG1);CHKERRQ(ierr);
  c->dirichlet = vD;
  ierr = VecDuplicate(uG, &c->force);CHKERRQ(ierr);
  *global = uG;
  ierr = VecGetArray(c->workV[0], &v);CHKERRQ(ierr);
  ierr = VecGetArray(c->dirichlet, &w);CHKERRQ(ierr);
  for (int i=0; i < dv; i++) w[i] = v[i];
  ierr = VecRestoreArray(c->workV[0], &v);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->dirichlet, &w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesCreateExactSolution"
PetscErrorCode StokesCreateExactSolution(SNES snes, Vec U, Vec U2)
{
  StokesCtx      *c;
  PetscInt        d, *dim;
  PetscReal      *v, *v2, *p, *p2, *coord;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes, (void **)&c);CHKERRQ(ierr);
  d = c->numDims; dim = c->dim;
  ierr = VecGetArray(c->workV[0], &v);CHKERRQ(ierr);
  ierr = VecGetArray(c->workV[1], &v2);CHKERRQ(ierr);
  ierr = VecGetArray(c->workP[0], &p);CHKERRQ(ierr);
  ierr = VecGetArray(c->workP[1], &p2);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord);CHKERRQ(ierr);
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    const PetscInt i = it.i;
    PetscReal u[d+1], u2[d+1];
    ierr = c->options->exact(d, &coord[i*d], u, u2, c->options->exactCtx);CHKERRQ(ierr);
    for (int j=0; j < d; j++) {
      v[i*d+j] = u[j];
      v2[i*d+j] = u2[j];
    }
    p[i] = u[d];
    p2[i] = u2[d];
  }
  ierr = VecRestoreArray(c->workV[0], &v);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workV[1], &v2);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workP[0], &p);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->workP[1], &p2);CHKERRQ(ierr);
  if (c->options->debug > 1) {
    ierr = VecPrint2(c->coord,    c->dim[0], c->dim[1], c->numDims, "coordinates",   "xy");CHKERRQ(ierr);
    ierr = VecPrint2(c->workV[0], c->dim[0], c->dim[1], c->numDims, "exact velocity",   "uv");CHKERRQ(ierr);
    ierr = VecPrint2(c->workP[0], c->dim[0], c->dim[1],          1, "exact pressure",   "p");CHKERRQ(ierr);
    ierr = VecPrint2(c->workV[1], c->dim[0], c->dim[1], c->numDims, "exact forcing",    "uv");CHKERRQ(ierr);
    ierr = VecPrint2(c->workP[1], c->dim[0], c->dim[1],          1, "exact divergence", "p");CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(c->scatterLP, c->workP[0], c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLP, c->workP[0], c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLV, c->workV[0], c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLV, c->workV[0], c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLP, c->workP[1], c->pG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLP, c->workP[1], c->pG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLV, c->workV[1], c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLV, c->workV[1], c->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->pG0, U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG0, U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG0, U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG0, U, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->pG1, U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG1, U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, U2, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecCopy(U2, c->force);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRemoveConstantPressure"
PetscErrorCode StokesRemoveConstantPressure(KSP ksp, StokesCtx *ctx, Vec *X, MatNullSpace *ns)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(ctx->pG0, 1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(ctx->force, X);CHKERRQ(ierr);
  ierr = VecZeroEntries(*X);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterPG, ctx->pG0, *X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPG, ctx->pG0, *X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecNormalize(*X, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ksp, &comm);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, 1, X, ns);CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ksp, *ns);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL, &ctx->NullSpaceSchur);CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ctx->KSPSchur, ctx->NullSpaceSchur);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPressureReduceOrder"
PetscErrorCode StokesPressureReduceOrder(Vec pL, StokesCtx *c)
{
  PetscInt d = c->numDims, *dim = c->dim;
  PetscReal *p, *coord, *work, *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (d != 2) SETERRQ1(1, "Not implemented for dimension %d\n", d);
  //ierr = VecPrint2(pL, c->dim[0], c->dim[1],          1, "unfiltered", "p");CHKERRQ(ierr);
  PetscInt m = dim[0], n = dim[1], mn = PetscMax(m,n);
  ierr = PetscMalloc2(4*mn, PetscReal, &work, mn, PetscReal, &x);CHKERRQ(ierr);
  ierr = VecGetArray(pL, &p);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord);CHKERRQ(ierr);
  for (int i=1; i < m; i++) {
    const PetscReal yM = coord[(i*n+0)*d+1];
    const PetscReal yP = coord[(i*n+n-1)*d+1];
    for (int j=0; j < n-2; j++) { // initialize Neville's algorithm
      x[j] = coord[(i*n+(j+1))*d+1]; // get `y` component
      work[j*4] = p[i*n+(j+1)];
      work[j*4+1] = work[j*4];
    }
    ierr = polyInterp(n-2, x, work, yM, yP, &p[i*n+0], &p[i*n+n-1]);CHKERRQ(ierr);
  }
  for (int j=0; j < n; j++) { // This will set the corner values as well.
    const PetscReal xM = coord[(0*n+j)*d];
    const PetscReal xP = coord[((m-1)*n+j)*d];
    for (int i=0; i < m-2; i++) {
      x[i]  = coord[((i+1)*n+j)*d+0]; // get `x' component
      work[i*4] = work[i*4+1] = p[(i+1)*n+j];
    }
    //for (int k=0; k < m-2; k++) { printf("%4f ", x[k]); } printf("\n");
    //for (int k=0; k < m-2; k++) { printf("%4f ", work[k*4]); } printf("\n");
    ierr = polyInterp(m-2, x, work, xM, xP, &p[0*n+j], &p[(m-1)*n+j]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pL, &p);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->coord, &coord);CHKERRQ(ierr);
  //ierr = VecPrint2(pL, c->dim[0], c->dim[1],          1, "filtered", "p");CHKERRQ(ierr);
  ierr = PetscFree2(work, x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPCApply"
PetscErrorCode StokesPCApply(void *void_ctx, Vec x, Vec y)
{
  StokesCtx *c = (StokesCtx *)void_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(c->scatterGV, x, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGV, x, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = KSPSolve(c->KSPVelocity, c->vG0, c->vG1);CHKERRQ(ierr);
  ierr = MatMult(c->MatPV, c->vG1, c->pG0);CHKERRQ(ierr);
  ierr = VecScale(c->pG0, -1.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGP, x, c->pG0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGP, x, c->pG0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = KSPSolve(c->KSPSchur, c->pG0, c->pG1);CHKERRQ(ierr);
  ierr = MatMult(c->MatVP, c->pG1, c->vG1);CHKERRQ(ierr);
  ierr = VecAXPY(c->vG0, -1.0, c->vG1);CHKERRQ(ierr);
  ierr = KSPSolve(c->KSPVelocity, c->vG0, c->vG1);CHKERRQ(ierr);

  ierr = VecScatterBegin(c->scatterPG, c->pG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRheologyLinear"
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx)
{

  PetscFunctionBegin; InFunction;
  *eta = 1.0; *deta = 0.0;
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
    rhs[0]   = PetscSqr(0.5 * PETSC_PI) * eta * u - 0.25 * PETSC_PI * sin(PETSC_PI * coord[0]) + 10;
    rhs[1]   = PetscSqr(0.5 * PETSC_PI) * eta * v - 0.25 * PETSC_PI * sin(PETSC_PI * coord[1]) + 10;
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
    rhs[0]   = PetscSqr(0.5 * PETSC_PI) * eta * u;
    rhs[1]   = PetscSqr(0.5 * PETSC_PI) * eta * v;
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
PetscErrorCode VecPrint2(Vec v, PetscInt m, PetscInt n, PetscInt F, const char *name, const char *component) {
  MPI_Comm comm;
  PetscScalar *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v, &comm); CHKERRQ(ierr);
  ierr = VecGetArray(v, &x); CHKERRQ(ierr);
  for (int f=0; f < F; f++) {
    for (int j=0; j<n; j++) {
      ierr = PetscPrintf(comm, "%14s %c: ", name, component[f]);CHKERRQ(ierr);
      for (int i=m-1; i >= 0; i--) {
        ierr = PetscPrintf(comm, "%12.3e", x[(i*n + j)*F + f]); CHKERRQ(ierr);
      }
      ierr = PetscPrintf(comm, "\n"); CHKERRQ(ierr);
    }
    if (f < F-1) { ierr = PetscPrintf(comm, "-----------\n"); CHKERRQ(ierr); }
  }
  ierr = VecRestoreArray(v, &x); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
