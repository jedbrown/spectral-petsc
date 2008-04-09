static char help[] = "Stokes problem with non-Newtonian rheology via Chebyshev collocation.\n";

//#define InFunction printf("%s\n", __FUNCT__)
#define InFunction
#define SOLVE 1
#define CHECK_EXACT 1

#include "chebyshev.h"
#include "util.C"
#include <petscsnes.h>
#include <stdbool.h>

typedef enum { DIRICHLET, NEUMANN, MIXED } BdyType;

typedef PetscErrorCode(*ExactSolution)(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
typedef PetscErrorCode(*BdyFunc)(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);
typedef PetscErrorCode(*Rheology)(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx);

typedef struct {
  ExactSolution  exact;
  void          *exactCtx;
} StokesExactBoundaryCtx;

typedef struct { // For higher dimensions, use a different value of `3'
  PetscReal normal[3], coord[3], value[3], alpha;
  BdyType   type;
  PetscInt  localIndex;
} StokesBoundaryMixed;

typedef struct {
  PetscInt       debug, cont0, cont;
  PetscReal      hardness, exponent, regularization, gamma0;
  Rheology       rheology;
  ExactSolution  exact;
  BdyFunc        boundary;
  void          *rheologyCtx, *exactCtx, *boundaryCtx;
} StokesOptions;

typedef struct {
  MPI_Comm       comm;
  StokesOptions *options;
  PetscInt       numDims, numWorkP, numWorkV, numMixed;
  PetscInt      *dim;
  KSP            KSPSchur, KSPVelocity, KSPSchurVelocity;
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
  Vec            massLump;
  IS             isLP, isPL, isLV, isVL, isLD, isDL, isPG, isGP, isVG, isGV; // perhaps useful for building preconditioner
  VecScatter     scatterLP, scatterPL; // pressure local <-> pressure global
  VecScatter     scatterLV, scatterVL; // velocity local <-> global
  VecScatter     scatterLD, scatterDL; // velocity local <-> special dirichlet
  VecScatter     scatterPG, scatterGP; // global pressure <-> full global
  VecScatter     scatterVG, scatterGV; // global velocity <-> full global
  StokesBoundaryMixed *mixed;
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
PetscErrorCode StokesMixedApply(StokesCtx *, Vec vL, Vec *stressL, Vec xL);
PetscErrorCode StokesMixedFilter(StokesCtx *, Vec xL);
PetscErrorCode StokesMixedVelocity(StokesCtx *c, Vec vL);
PetscErrorCode StokesPCApply(void *, Vec, Vec);
PetscErrorCode StokesPCSetUp(void *);
PetscErrorCode StokesPCSetUp1(void *);
PetscErrorCode StokesSchurMatMult(Mat, Vec, Vec);
PetscErrorCode StokesVelocityMatMult(Mat, Vec, Vec);
PetscErrorCode StokesStateView(StokesCtx *ctx, Vec state, const char *filename);
PetscErrorCode StokesVecView(Vec v, PetscInt nodes, PetscInt pernode, PetscInt perline, PetscViewer view);

PetscErrorCode VecPrint2(Vec v, PetscInt m, PetscInt n, PetscInt F, const char *name, const char *component);
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx);
PetscErrorCode StokesRheologyPower(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx);
PetscErrorCode StokesExact0(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesExact1(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesExact2(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesExact3(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesDirichlet(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);
PetscErrorCode StokesBoundary1(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);
PetscErrorCode StokesBoundary2(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  MPI_Comm             comm = PETSC_COMM_SELF;
  StokesCtx           *ctx;
  StokesOptions       *opt;
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
  opt = ctx->options;
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
  ierr = SNESSetJacobian(snes, A, P, StokesJacobian, ctx);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = StokesRemoveConstantPressure(ksp, ctx, &nv, &ns);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPFGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc, ctx);CHKERRQ(ierr);
  {
    ierr = PetscOptionsHasName(PETSC_NULL, "-prefd", &flag);CHKERRQ(ierr);
    ierr = PCShellSetSetUp(pc, flag ? StokesPCSetUp : StokesPCSetUp1);CHKERRQ(ierr);
  }
  ierr = PCShellSetApply(pc, StokesPCApply);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // u = exact solution, u2 = A(u) u (used as forcing term)
  ierr = StokesCreateExactSolution(snes, u, u2);CHKERRQ(ierr);
  ierr = StokesFunction(snes, u, r, ctx);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, &unorm);CHKERRQ(ierr);
  ierr = VecNorm(u2, NORM_INFINITY, &u2norm);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_INFINITY, &rnorm);CHKERRQ(ierr);
  {
    ierr = PetscPrintf(comm, "Norm of solution %9.3e  norm of forcing %9.3e  norm of residual %9.3e\n", unorm, u2norm, rnorm);CHKERRQ(ierr);
    if (opt->debug > 0) {
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
    PetscReal exponent = opt->exponent;
    ierr = VecSet(x, 0.0);CHKERRQ(ierr);
    for (int i = opt->cont0; i < opt->cont+1; i++) {
      opt->exponent = 1.0 + pow(1.0*i/opt->cont, 0.8) * (exponent - 1.0);
      ierr = PetscPrintf(comm, "## [%d/%d] Solving with exponent = %f\n", i, opt->cont, opt->exponent);CHKERRQ(ierr);
      ierr = SNESSolve(snes, PETSC_NULL, x);CHKERRQ(ierr);
      ierr = VecCopy(x, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
      ierr = MatNullSpaceRemove(ns, r, PETSC_NULL);CHKERRQ(ierr);
      if (opt->debug > 0) {
        ierr = VecPrint2(x, ctx->dim[0]-2, ctx->dim[1]-2, 3, "final error", "uvp");CHKERRQ(ierr);
      }
      ierr = VecNorm(r, NORM_INFINITY, &norm);CHKERRQ(ierr);
      ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
      ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Number of nonlinear iterations = %d\n", its);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "%-25s: abs = %8e\n", "Norm of error", norm);CHKERRQ(ierr);
    }
  }
  ierr = StokesStateView(ctx, x, "final state");CHKERRQ(ierr);

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
  c->numWorkP = 2*d+1; c->numWorkV = 2 + 3*d;
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
        x[it.i*d+j] = cos(it.ind[j] * PETSC_PI / (dim[j] - 1)); // Chebyshev spacing
        //x[it.i*d+j] = 1 - 2.0 * it.ind[j] / (dim[j] - 1); // linear spacing
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
    ierr = VecDuplicate(c->vG0, &c->massLump);CHKERRQ(ierr);
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
    ierr = KSPCreate(comm, &c->KSPSchurVelocity);CHKERRQ(ierr);
    ierr = KSPSetOperators(c->KSPSchurVelocity, c->MatVV, c->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(c->KSPSchurVelocity, "svel_");CHKERRQ(ierr);
    ierr = KSPSetFromOptions(c->KSPSchurVelocity);CHKERRQ(ierr);
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
  ierr = KSPDestroy(c->KSPSchurVelocity);CHKERRQ(ierr);
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
  ierr = VecDestroy(c->massLump);CHKERRQ(ierr);
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
  if (c->numMixed > 0) { ierr = PetscFree(c->mixed);CHKERRQ(ierr); }
  ierr = PetscFree2(c->DP, c->DV);CHKERRQ(ierr);
  ierr = PetscFree(c->dim);CHKERRQ(ierr);
  if (c->options->boundaryCtx) { ierr = PetscFree(c->options->boundaryCtx);CHKERRQ(ierr); }
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
  opt->debug = 0; opt->hardness = 1.0; opt->exponent = 1.0; opt->regularization = 1.0; opt->gamma0 = 1.0; opt->cont0 = 0; opt->cont = 1;
  ierr = PetscOptionsBegin(comm, "", "Stokes problem options", "");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dim", "list of dimension extent", "stokes.C", ctx->dim, &ctx->numDims, &flag);CHKERRQ(ierr);
  if (!flag) { ctx->numDims = 2; ctx->dim[0] = 8; ctx->dim[1] = 6; }
  ierr = PetscOptionsInt("-debug", "debugging level", "stokes.C", opt->debug, &opt->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-exact", "exact solution type", "stokes.C", exact, &exact, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-boundary", "boundary type", "stokes.C", boundary, &boundary, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rheology", "rheology type", "stokes.C", rheology, &rheology, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hardness", "power law hardness parameter", "stokes.C", opt->hardness, &opt->hardness, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-exponent", "power law exponent", "stokes.C", opt->exponent, &opt->exponent, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps", "regularization parameter for viscosity", "stokes.C", opt->regularization, &opt->regularization, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-gamma0", "reference strain", "stokes.C", opt->gamma0, &opt->gamma0, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-cont0", "starting continuation", "stokes.C", opt->cont0, &opt->cont0, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-cont", "number of continuations", "stokes.C", opt->cont, &opt->cont, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscPrintf(comm, "Stokes problem");CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  dim = [", ctx->numDims);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    if (i > 0) { ierr = PetscPrintf(comm, ",");CHKERRQ(ierr); }
    ierr = PetscPrintf(comm, "%d", ctx->dim[i]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm, "]\n  hardness = %f    exponent = %8f    regularization = %8f    gamma0 = %8f\n",
                     opt->hardness, opt->exponent, opt->regularization, opt->gamma0);CHKERRQ(ierr);

  switch (exact) {
    case 0:
      opt->exact    = StokesExact0;
      opt->exactCtx = PETSC_NULL;
      break;
    case 1:
      opt->exact    = StokesExact1;
      opt->exactCtx = PETSC_NULL;
      break;
    case 2:
      opt->exact    = StokesExact2;
      opt->exactCtx = PETSC_NULL;
      break;
    case 3:
      opt->exact    = StokesExact3;
      opt->exactCtx = PETSC_NULL;
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Exact solution %d not implemented", exact);
  }
  switch (boundary) {
    case 0:
      opt->boundary    = StokesDirichlet; // The dirichlet condition just evaluates the exact solution.
      ierr = PetscMalloc(sizeof(StokesExactBoundaryCtx), &opt->boundaryCtx);CHKERRQ(ierr);
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exact = opt->exact;
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exactCtx = opt->exactCtx;
      break;
    case 1:
      opt->boundary    = StokesBoundary1; // This condition evaluates and numerically differentiates the exact solution;
      ierr = PetscMalloc(sizeof(StokesExactBoundaryCtx), &opt->boundaryCtx);CHKERRQ(ierr);
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exact = opt->exact;
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exactCtx = opt->exactCtx;
      break;
    case 2:
      opt->boundary    = StokesBoundary2; // This condition evaluates and numerically differentiates the exact solution;
      ierr = PetscMalloc(sizeof(StokesExactBoundaryCtx), &opt->boundaryCtx);CHKERRQ(ierr);
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exact = opt->exact;
      ((StokesExactBoundaryCtx *)opt->boundaryCtx)->exactCtx = opt->exactCtx;
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Boundary type %d not implemented", exact);
  }
  switch (rheology) {
    case 0:
      opt->rheology    = StokesRheologyLinear;
      opt->rheologyCtx = PETSC_NULL;
      break;
    case 1:
      opt->rheology    = StokesRheologyPower;
      opt->rheologyCtx = (void *)opt;
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
  ierr = KSPSolve(ctx->KSPSchurVelocity, ctx->vG0, ctx->vG1);CHKERRQ(ierr);
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
  ierr = StokesMixedVelocity(ctx, ctx->workV[0]);CHKERRQ(ierr);
  if (withDirichlet) {
    ierr = VecScatterBegin(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  }
  //ierr = VecView(ctx->dirichlet, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->workP[2]);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    ierr = VecStrideGather(ctx->workV[0], i, ctx->workP[0], INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    // FIXME: coordinate transform
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
  for (int i=0; i < ctx->numDims; i++) { // FIXME: coordinate transform
    ierr = MatMult(ctx->DP[i], ctx->workP[0], ctx->workP[1]);CHKERRQ(ierr);
    ierr = VecStrideScatter(ctx->workP[1], i, ctx->workV[0], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = StokesMixedFilter(ctx, ctx->workV[0]);CHKERRQ(ierr);
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
  Vec             xL, yL, *V, *W, *Stress;
  PetscReal      *eta, *deta, **v, **Strain, **stress;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&ctx);CHKERRQ(ierr);
  d = ctx->numDims; dim = ctx->dim; n = productInt(d, dim);
  xL = ctx->workV[0]; yL = ctx->workV[1]; V = &ctx->workV[2]; W = &ctx->workV[2+d]; Stress = &ctx->workV[2+2*d];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesMixedVelocity(ctx, xL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(ctx->DV[i], xL, V[i]);CHKERRQ(ierr); }
  // FIXME: Coordinate transformation
  ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta);CHKERRQ(ierr);
  ierr = VecGetArrays(ctx->strain, d, &Strain);CHKERRQ(ierr);
  ierr = VecGetArrays(Stress, d, &stress);CHKERRQ(ierr);
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
        const PetscReal s = eta[i] * strain[j][k];
        stress[j][i*d+k] = s;
        v[j][i*d+k] = s + deta[i] * Strain[j][i*d+k] * z;
      }
    }
  }
  ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->eta, &eta);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->deta, &deta);CHKERRQ(ierr);
  ierr = VecRestoreArrays(ctx->strain, d, &Strain);CHKERRQ(ierr);
  ierr = VecRestoreArrays(Stress, d, &stress);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = MatMult(ctx->DV[i], V[i], W[i]);CHKERRQ(ierr); }
  // FIXME: coordinate transform
  ierr = VecZeroEntries(yL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = VecAXPY(yL, -1.0, W[i]);CHKERRQ(ierr); }
  ierr = StokesMixedApply(ctx, xL, Stress, yL);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterLV, yL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterLV, yL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesFunction"
PetscErrorCode StokesFunction(SNES snes, Vec xG, Vec yG, void *ctx)
{
  StokesCtx       *c = (StokesCtx *)ctx;
  PetscInt         n, d;
  Vec              xL, yL, *V, *W;
  PetscReal      **v, **strain, *eta, *deta;
  PetscErrorCode   ierr;

  PetscFunctionBegin; InFunction;
  d = c->numDims; n = productInt(d, c->dim);
  xL = c->workV[0]; yL = c->workV[1]; V = &c->workV[2]; W = &c->workV[2+d];
  ierr = VecScatterBegin(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGP, xG, c->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGV, xG, c->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVL, c->vG0, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVL, c->vG0, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesMixedVelocity(c, xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  for (int i=0; i < d; i++) { ierr = MatMult(c->DV[i], xL, c->strain[i]);CHKERRQ(ierr); }
  // FIXME: coordinate tranform

  if (true) { // Symmetrize strain, compute nonlinear contributions
    PetscReal s[d][d], gamma;
    ierr = VecGetArrays(c->strain, d, &strain);CHKERRQ(ierr);
    ierr = VecGetArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecGetArray(c->deta, &deta);CHKERRQ(ierr);
    ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      gamma = 0.0;
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
    {
      PetscReal minEta, maxEta;
      ierr = VecMin(c->eta, PETSC_NULL, &minEta);CHKERRQ(ierr);
      ierr = VecMax(c->eta, PETSC_NULL, &maxEta);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Minimum eta = %9.3e   Maximum eta = %9.3e\n", minEta, maxEta);CHKERRQ(ierr);
    }
  }
  for (int i=0; i < d; i++) { ierr = MatMult(c->DV[i], V[i], W[i]);CHKERRQ(ierr); }
  // FIXME: coordinate transform
  ierr = VecZeroEntries(yL);CHKERRQ(ierr);
  for (int i=0; i < d; i++) { ierr = VecAXPY(yL, -1.0, W[i]);CHKERRQ(ierr); }
  ierr = StokesMixedApply(c, xL, V, yL);CHKERRQ(ierr);
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
PetscErrorCode StokesJacobian(SNES snes, Vec w, Mat *Ashell, Mat *Pshell, MatStructure *flag, void *void_ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // The nonlinear term has already been fixed up by StokesFunction() so we do nothing here.
  *flag = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesSetupDomain"
PetscErrorCode StokesSetupDomain(StokesCtx *c, Vec *global)
{
  MPI_Comm             comm = c->comm;
  PetscInt             d    = c->numDims, *dim = c->dim, m = productInt(d, dim), n = d*m, N=n+m;
  StokesBoundaryMixed *mixed;
  IS                   isG, isD;
  PetscInt            *ixLP, *ixPL, *ixLV, *ixVL, *ixLD, *ixDL, *ixPG, *ixGP, *ixVG, *ixGV;
  PetscInt             lp, lv, gp, gv, dv, g, im;
  PetscReal           *v, *w, *x;
  BdyType              type;
  Vec                  uG, pL, pG, vL, vG, vD;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc6(m, PetscInt, &ixLP, m, PetscInt, &ixPL, n, PetscInt, &ixLV, n, PetscInt, &ixVL, n, PetscInt, &ixLD, n, PetscInt, &ixDL);CHKERRQ(ierr);
  ierr = PetscMalloc5(m, PetscInt, &ixPG, N, PetscInt, &ixGP, n, PetscInt, &ixVG, N, PetscInt, &ixGV, m, StokesBoundaryMixed, &mixed);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);    // Coordinates in a block-size d vector
  ierr = VecGetArray(c->workV[0], &v);CHKERRQ(ierr); // Some workspace for boundary values
  lp=lv=gp=gv=dv=g=im=0;
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    PetscReal normal[d];
    if (it.normal(normal)) { // On the boundary
      ierr = c->options->boundary(d, &x[it.i*d], normal, &type, &v[dv], c->options->boundaryCtx);CHKERRQ(ierr);
      switch (type) {
        case DIRICHLET:
          for (int k=0; k < d; k++) { // velocity in the local system comes from dirichlet values
            ixLV[lv] = -1;                            // local dof is not in the global system
            ixDL[dv] = lv; ixLD[lv] = dv; lv++; dv++; // 2-way mapping from local velocity to dirichlet
          }
          break;
        case NEUMANN: // We keep all the mixed conditions together, grouped by node.
          mixed[im].type       = NEUMANN;
          mixed[im].localIndex = it.i;
          mixed[im].alpha      = 0.0;
          for (int k=0; k < d; k++) { mixed[im].coord[k] = x[it.i*d+k]; mixed[im].normal[k] = normal[k]; mixed[im].value[k] = v[dv+k]; }
          if (c->options->debug > 1) {
            printf("boundary type NEUMANN, localIndex = %d\n", mixed[im].localIndex);
            ierr = PetscRealView(d, mixed[im].coord, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            ierr = PetscRealView(d, mixed[im].normal, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            ierr = PetscRealView(d, mixed[im].value, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          }
          im++;
          // Velocity is just like an interior point.  This loop is *copied* from below.
          for (int k=0; k < d; k++) {
            ixLV[lv] = gv; ixVL[gv] = lv; // two way map from local to global velocity
            ixVG[gv] = g;  ixGV[g ] = gv; // two way map from global velocity to full global
            ixGP[g] = -1;
            ixLD[lv] = -1;
            lv++; gv++; g++;
          }
          break;
        case MIXED:
          // The boundary function returns the sliding coefficient in the first element of 'value' and the extra traction in the next 'd' entries.
          mixed[im].type       = MIXED;
          mixed[im].localIndex = it.i;
          mixed[im].alpha      = v[dv];
          for (int k=0; k < d; k++)   { mixed[im].coord[k] = x[it.i*d+k]; mixed[im].normal[k] = normal[k]; mixed[im].value[k] = v[dv+k+1]; }
          if (c->options->debug > 1) {
            printf("boundary type MIXED, localIndex = %d, alpha = %8g\n", mixed[im].localIndex, mixed[im].alpha);
            ierr = PetscRealView(d, mixed[im].coord, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            ierr = PetscRealView(d, mixed[im].normal, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            ierr = PetscRealView(d, mixed[im].value, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
          }
          im++;
          { // The velocity component 'most in the normal direction' is removed from the system, the rest are treated as usual
            const PetscInt in = indexMaxAbs(d, normal);
            for (int k=0; k < d; k++) {
              if (k == in) { // component is not in global system
                ixLV[lv] = -1;
                ixLD[lv] = -1;
                lv++;
              } else { // component is in global system
                ixLV[lv] = gv; ixVL[gv] = lv; // two way map from local to global velocity
                ixVG[gv] = g;  ixGV[g ] = gv; // two way map from global velocity to full global
                ixGP[g] = -1;
                ixLD[lv] = -1;
                lv++; gv++; g++;
              }
            }
          }
          break;
        default:
          SETERRQ(1, "Boundary type not implemented.");
      }
      ixLP[lp++] = -1; // local pressure at the boundary is never represented in global pressure
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
  c->numMixed = im;
  if (im > 0) {
    ierr = PetscMalloc(im*sizeof(StokesBoundaryMixed), &c->mixed);CHKERRQ(ierr);
    ierr = PetscMemcpy(c->mixed, mixed, im*sizeof(StokesBoundaryMixed));CHKERRQ(ierr);
  }
  ierr = VecCreate(comm, &uG);CHKERRQ(ierr);   ierr = VecSetSizes(uG,  g, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(uG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &pG);CHKERRQ(ierr);   ierr = VecSetSizes(pG, gp, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(pG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vG);CHKERRQ(ierr);   ierr = VecSetSizes(vG, gv, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(vG);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vD);CHKERRQ(ierr);   ierr = VecSetSizes(vD, dv, PETSC_DECIDE);CHKERRQ(ierr);   ierr = VecSetFromOptions(vD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "DOF distribution: %d global   %d/%d pressure    %d/%d velocity    %d dirichlet    %d mixed\n", g, gp, lp, gv, lv, dv, im);CHKERRQ(ierr);
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
  ierr = PetscFree5(ixPG, ixGP, ixVG, ixGV, mixed);CHKERRQ(ierr);
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
  for (int i=0; i < c->numMixed; i++) {
    for (int j=0; j < d; j++) {
      const PetscInt I = c->mixed[i].localIndex;
      v2[I*d+j] = c->mixed[i].value[j];
    }
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
  PetscInt        d = c->numDims, *dim = c->dim, m, n, p, mnp;
  PetscReal      *pres, *coord, *work, *x;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (d > 3) SETERRQ1(1, "Not implemented for dimension %d\n", d);
  //ierr = VecPrint2(pL, c->dim[0], c->dim[1],          1, "unfiltered", "p");CHKERRQ(ierr);
  m = dim[0]; n = dim[1]; p = (d == 2) ? 1 : dim[2]; mnp = PetscMax(PetscMax(m,n),p);
  ierr = PetscMalloc2(4*mnp, PetscReal, &work, mnp, PetscReal, &x);CHKERRQ(ierr);
  ierr = VecGetArray(pL, &pres);CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord);CHKERRQ(ierr);
  for (int i=1; i < m; i++) { // extend in the 'z' direction
    if (p > 1) { // This part only applies in 3D
      for (int j=1; j < n; j++) {
        const PetscInt  iM = (i*n+j)*p+0,   iP = (i*n+j)*p+p-1;
        const PetscReal zM = coord[iM*d+2], zP = coord[iP*d+2];
        for (int k=1; k < p-1; k++) {
          x[k-1] = coord[(iM+k)*d+2]; // get 'z' component
          work[(k-1)*4] = work[(k-1)*4+1] = pres[iM+k];
        }
        ierr = polyInterp(p-2, x, work, zM, zP, &pres[iM], &pres[iP]);CHKERRQ(ierr);
      }
    }
    for (int k=0; k < p; k++) {
      const PetscInt  iM = (i*n+0)*p+k,   iP = (i*n+n-1)*p+k;
      const PetscReal yM = coord[iM*d+1], yP = coord[iP*d+1];
      for (int j=1; j < n-1; j++) {
        x[j-1] = coord[(iM+j*p)*d+1];
        work[(j-1)*4] = work[(j-1)*4+1] = pres[iM+j*p];
      }
      ierr = polyInterp(n-2, x, work, yM, yP, &pres[iM], &pres[iP]);CHKERRQ(ierr);
    }
  }
  for (int j=0; j < n; j++) {
    for (int k=0; k < p; k++) {
      const PetscInt  iM = (0*n+j)*p+k,   iP = ((m-1)*n+j)*p+k;
      const PetscReal xM = coord[iM*d+0], xP = coord[iP*d+0];
      for (int i=1; i < m-1; i++) {
        x[i-1] = coord[(iM+i*n*p)*d+0];
        work[(i-1)*4] = work[(i-1)*4+1] = pres[iM+i*n*p];
      }
      ierr = polyInterp(m-2, x, work, xM, xP, &pres[iM], &pres[iP]);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(pL, &pres);CHKERRQ(ierr);
  ierr = VecRestoreArray(c->coord, &coord);CHKERRQ(ierr);
  //ierr = VecPrint2(pL, c->dim[0], c->dim[1],          1, "filtered", "p");CHKERRQ(ierr);
  ierr = PetscFree2(work, x);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMixedApply"
PetscErrorCode StokesMixedApply(StokesCtx *c, Vec vL, Vec *stressL, Vec xL)
{
  const PetscInt  d = c->numDims;
  PetscReal      *x, *v, **stress;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(vL, &v);CHKERRQ(ierr);
  ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
  ierr = VecGetArrays(stressL, d, &stress);CHKERRQ(ierr);
  for (int i=0; i < c->numMixed; i++) {
    const PetscInt I = c->mixed[i].localIndex;
    for (int j=0; j < d; j++) {
      PetscReal z = 0.0;
      for (int k=0; k < d; k++) {
        z += stress[j][I*d+k] * c->mixed[i].normal[k];
      }
      x[I*d+j] = z + c->mixed[i].alpha * v[i*d+j];
    }
  }
  ierr = VecGetArray(vL, &v);CHKERRQ(ierr);
  ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
  ierr = VecGetArrays(stressL, d, &stress);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMixedFilter"
PetscErrorCode StokesMixedFilter(StokesCtx *c, Vec xL)
{
  const PetscInt  d = c->numDims;
  PetscReal      *x;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
  for (int i=0; i < c->numMixed; i++) {
    const PetscInt I = c->mixed[i].localIndex;
    for (int j=0; j < d; j++) {
      x[I*d+j] = 0.0;
    }
  }
  ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesMixedVelocity"
// With MIXED boundary, the component 'most parallel to the normal' is removed
// from the global system.  This function recovers that value so that there is
// no flux through the boundary.
PetscErrorCode StokesMixedVelocity(StokesCtx *c, Vec vL)
{
  const PetscInt  d = c->numDims;
  PetscReal      *v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(vL, &v);CHKERRQ(ierr);
  for (int i=0; i < c->numMixed; i++) {
    if (c->mixed[i].type == MIXED) {
      const PetscReal *normal = c->mixed[i].normal;
      const PetscInt in = indexMaxAbs(d, normal);
      PetscReal *vel = &v[i*d];
      vel[in] = 0.0;
      v[i*d+in] = -dotScalar(d, vel, normal) / normal[in];
    }
  }
  ierr = VecRestoreArray(vL, &v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPCSetUp"
PetscErrorCode StokesPCSetUp(void *void_ctx)
{
  StokesCtx     *ctx = (StokesCtx *)void_ctx;
  PetscInt       d = ctx->numDims, *dim = ctx->dim;
  PetscReal     *x, *eta, *deta, **strain;
  PetscInt      *ixL;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArrays(ctx->strain, d, &strain); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISGetIndices(ctx->isLV, &ixL); CHKERRQ(ierr);
  {
    PetscInt row, col[2*d+1], c, im=0;
    PetscScalar v[2*d+1];
    PetscScalar x0, xMM, xPP, xM, idxM, xP, idxP, idx, eM, deM, du0M, eP, deP, du0P;
    for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) { // loop over local dof
      const PetscInt i = it.i;
      if (im < ctx->numMixed && i == ctx->mixed[im].localIndex) { // we are at a mixed boundary node
        // for (int f=0; f < d; f++) { // Just put 1 on the diagonal.  That is, don't enforce the boundary condition in the preconditioner.
        //   row = ixL[i*d+f]; col[0] = row; v[0] = 1.0;
        //   ierr = MatSetValues(ctx->MatVVPC, 1, &row, 1, col, v, INSERT_VALUES); CHKERRQ(ierr);
        // }
        for (int f=0; f < d; f++) {
          //PetscInt j=0, pm=0, z=0;
          // for (int k=0; k < d; k++) {
          //   PetscReal tmp = ctx->mixed[im].normal[k];
          //   if (PetscSqr(tmp) > z) {
          //     z = tmp;
          //     j = k;
          //     pm = (tmp > 0) ? 1 : -1; // Chebyshev ordering
          //   }
          // }
          const PetscReal *normal = ctx->mixed[im].normal;
          const PetscInt j = indexMaxAbs(d, ctx->mixed[im].normal);
          const PetscInt pm = (normal[j] > 0) ? 1 : -1;
          const PetscInt iM = it.shift(j, pm); // Step in the principle normal direction.
          x0 = x[i*d+j]; xM = x[iM*d+j]; idx = 1.0 / (x0 - xM);
          row = ixL[i*d+f]; col[0] = row; col[1] = ixL[iM*d+f];
          v[0] = idx * eta[i];
          v[1] = -idx * eta[i];
          if (ctx->mixed[im].type == MIXED) {
            v[0] += ctx->mixed[im].alpha;
          }
          ierr = MatSetValues(ctx->MatVVPC, 1, &row, 2, col, v, INSERT_VALUES);CHKERRQ(ierr);
        }
        im++;
      } else { // We are at an interior or Dirichlet node
        for (int f=0; f < d; f++) { // Each equation, corresponds to a row in the matrix
          if (ixL[i*d+f] < 0) continue; // Not a global dof
          row = ixL[i*d+f]; col[0] = row; v[0] = 0.0; c=1; // initialize diagonal term
          for (int j=0; j < d; j++) { // each direction
            const PetscInt iM = it.shift(j, -1);
            const PetscInt iP = it.shift(j,  1);
            if (iM < 0 || iP < 0) SETERRQ(1, "Local neighbor not on local grid.");
            x0 = x[i*d+j]; xMM = x[iM*d+j]; xPP = x[iP*d+j];
            xM = 0.5*(xMM+x0); idxM = 1.0/(x0-xMM); xP = 0.5*(x0+xPP); idxP = 1.0/(xPP-x0); idx = 1.0/(xP-xM);
            eM = 0.5*(eta[iM]+eta[i]); deM = 0.5*(deta[iM]+deta[i]);
            eP = 0.5*(eta[iP]+eta[i]); deP = 0.5*(deta[iP]+deta[i]);
            //printf("eta %f %f %f\n", eta[iM], eta[i], eta[iP]);
            col[c] = ixL[iM*d+f]; v[c] = -idx * (idxM * eM); c++;
            col[c] = ixL[iP*d+f]; v[c] = -idx * (idxP * eP); c++;
            v[0] += idx * (idxP * eP + idxM * eM);
          }
          ierr = MatSetValues(ctx->MatVVPC, 1, &row, c, col, v, INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecRestoreArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecRestoreArrays(ctx->strain, d, &strain); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ctx->isLV, &ixL); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPVelocity, ctx->MatVV, ctx->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPSchurVelocity, ctx->MatVV, ctx->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPSchur, ctx->MatSchur, ctx->MatSchur, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  //ierr = MatView(ctx->MatVVPC, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesPCSetUp1"
PetscErrorCode StokesPCSetUp1(void *void_ctx)
{
#define ORDER 3
#if (ORDER == 2)
  const PetscReal qpoint[2]   = { -0.57735026918962573, 0.57735026918962573 };
  const PetscReal qweight[2]  = { 1.0, 1.0 };
  const PetscReal basis[2][2] = {{0.78867513459481287, 0.21132486540518708}, {0.21132486540518708, 0.78867513459481287}};
  const PetscReal deriv[2][2] = {{-0.5, -0.5}, {0.5, 0.5}};
  const PetscInt  qdim[]      = {2,2,2,2,2}; // 2 quadrature points in each direction
#elif (ORDER == 3)
  const PetscReal qpoint[3]   = {-0.7745966692414834, -2.4651903288156619e-31, 0.7745966692414834};
  const PetscReal qweight[3]  = {0.55555555555556, 0.88888888888889, 0.55555555555556};
  const PetscReal basis[2][3] = {{0.887298334621,0.5,0.112701665379},{0.112701665379,0.5,0.887298334621}};
  const PetscReal deriv[2][3] = {{-0.5, -0.5, -0.5},{0.5, 0.5, 0.5}};
  const PetscInt  qdim[]      = {3,3,3,3,3};
#endif
  const PetscInt ndim[]       = {2,2,2,2,2}; // 2 nodes in each direction within each element
  StokesCtx      *ctx         = (StokesCtx *)void_ctx;
  const PetscInt d = ctx->numDims, *dim = ctx->dim, N = productInt(d, ndim);
  PetscReal      *x, *eta, *deta, **strain, *lump;
  PetscInt       *ixL;
  PetscInt row[N*d], col[N*d];
  PetscReal A[N*d][N*d], M[N*d][N*d];
  PetscTruth      flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(ctx->massLump);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->massLump, &lump);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArrays(ctx->strain, d, &strain); CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISGetIndices(ctx->isLV, &ixL); CHKERRQ(ierr);
  ierr = MatZeroEntries(ctx->MatVVPC);CHKERRQ(ierr);
  for (BlockIt el = BlockIt(d, dim); !el.done; el.next()) { // loop over elements
    bool skip = false;
    for (int i=0; i < d; i++) { if (el.ind[i] == dim[i]-1) { skip = true; } }
    if (skip) continue;
    // compute element jacobian at quadrature points
    PetscReal J[d][d], Jinv[d][d], Jdet;
    for (int i=0; i < d; i++) { // direction in reference cell
      const PetscInt iP = el.shift(i, 1);
      for (int j=0; j < d; j++) { // derivative direction
        J[i][j] = 0.5 * (x[iP*d+j] - x[el.i*d+j]);
      }
    }
    {
      if (d != 2) SETERRQ1(1, "Jacobian inverse not implemented for dimension %d", d);
      Jdet = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      const PetscReal iJdet = 1.0 / Jdet;
      Jinv[0][0] =  iJdet * J[1][1];     Jinv[0][1] = -iJdet * J[0][1];
      Jinv[1][0] = -iJdet * J[1][0];     Jinv[1][1] =  iJdet * J[0][0];
    }
    ierr = PetscMemzero(row, N*d*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(col, N*d*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(A, PetscSqr(N*d)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(M, PetscSqr(N*d)*sizeof(PetscReal));CHKERRQ(ierr);
    for (BlockIt quad = BlockIt(d, qdim); !quad.done; quad.next()) {
      PetscReal qw = Jdet; //qw = 1.0;
      // The finite element formulation needs the Jacobian determinant here, but the preconditioner is stronger when we
      // just use a constant.  Presumably this is due to the collocation nature of the spectral method.  Alternatively,
      // we can invert the mass matrix which corrects the scaling.  However, lumping the mass matrix seems to be
      // equivalent to just ignoring the determinant here.
      for (int i=0; i < d; i++) qw *= qweight[quad.ind[i]];
      for (BlockIt test = BlockIt(d, ndim); !test.done; test.next()) {
        for (int a=0; a < d; a++) { // test function component
          row[test.i*d+a] = ixL[el.plus(test.ind)*d+a];
          for (BlockIt trial = BlockIt(d, ndim); !trial.done; trial.next()) {
            for (int b=0; b < d; b++) { // trial function component
              col[trial.i*d+b] = ixL[el.plus(trial.ind)*d+b];
              PetscReal D[d][d], E[d][d], dtest[d], dtrial[d];
              ierr = PetscMemzero(D, d*d*sizeof(PetscReal));CHKERRQ(ierr);
              ierr = PetscMemzero(E, d*d*sizeof(PetscReal));CHKERRQ(ierr);
              ierr = PetscMemzero(dtest, d*sizeof(PetscReal));CHKERRQ(ierr);
              ierr = PetscMemzero(dtrial, d*sizeof(PetscReal));CHKERRQ(ierr);
              for (int i=0; i < d; i++) { // real derivative direction
                for (int j=0; j < d; j++) { // reference derivative direction
                  PetscReal ztest=1.0, ztrial=1.0;
                  for (int k=0; k < d; k++) { // tensor product direction
                    if (j == k) {
                      ztest  *= deriv[ test.ind[j]][quad.ind[j]] * Jinv[j][i];
                      ztrial *= deriv[trial.ind[j]][quad.ind[j]] * Jinv[j][i];
                    } else {
                      ztest  *= basis[ test.ind[k]][quad.ind[k]];
                      ztrial *= basis[trial.ind[k]][quad.ind[k]];
                    }
                  }
                  dtest[i] += ztest;
                  dtrial[i] += ztrial;
                }
              }
              for (int i=0; i < d; i++) { // derivative direction
                E[a][i] += 0.5 * dtest[i];
                E[i][a] += 0.5 * dtest[i];
                D[b][i] += 0.5 * dtrial[i];
                D[i][b] += 0.5 * dtrial[i];
              }
              PetscReal z=0.0, zhat=0.0, zz=0.0;
              for (int i=0; i < d; i++) {
                for (int j=0; j < d; j++) {
                  z    += E[i][j] * D[i][j];
                  zhat += E[i][j] * strain[j][el.i*d+i];
                  zz   += D[i][j] * strain[j][el.i*d+i];
                }
              }
              if (false) { // Just the laplacian
                z = 0.0;
                if (a == b) {
                  for (int i=0; i < d; i++) z += dtest[i] * dtrial[i];
                  //z += dtest[0] * dtrial[0];
                }
                A[test.i*d+a][trial.i*d+b] += (eta[el.i] * z + deta[el.i] * zhat * zz) * qw;
              } else { // The full system
                A[test.i*d+a][trial.i*d+b] += (eta[el.i] * z + deta[el.i] * zhat * zz) * qw;
              }
              PetscReal zmass = 1.0;
              for (int i=0; i < d; i++) { // tensor product direction
                zmass *= basis[ test.ind[i]][quad.ind[i]] * basis[trial.ind[i]][quad.ind[i]];
              }
              M[test.i*d+a][trial.i*d+b] += zmass * qw;
            }
          }
        }
      }
    }
    if (ctx->options->debug > 0) { // element matrix diagnostics
      PetscInt m=0, n=0;
      for (int i=0; i < d*N; i++) {
        if (row[i] >= 0) m++;
        if (col[i] >= 0) n++;
      }
      PetscReal Arelevant[m][n];
      ierr = PetscMemzero(Arelevant, m*n*sizeof(PetscReal));CHKERRQ(ierr);
      PetscInt I=0, J=0;
      for (int i=0; i < d*N; i++) {
        if (row[i] < 0) continue;
        J = 0;
        for (int j=0; j < d*N; j++) {
          if (col[j] < 0) continue;
          Arelevant[I][J] = A[i][j];
          J++;
        }
        I++;
      }
      printf("element %d,%d: col =", el.ind[0], el.ind[1]);
      for (int j=0; j < d*N; j++) {
        if (col[j] >= 0) { printf("%5d", col[j]); }
      }
      printf("\n");
      I = 0;
      for (int i=0; i < d*N; i++) {
        if (row[i] < 0) continue;
        printf("[%2d] ", row[i]);
        J=0;
        for (int j=0; j < d*N; j++) {
          if (col[j] < 0) continue;
          printf("%10.7f ", Arelevant[I][J]);
          J++;
        }
        I++;
        printf("\n");
      }
    }
    for (int i=0; i < N*d && true; i++) { // lump the element mass matrix
      if (row[i] < 0) continue;
      for (int j=0; j < N*d; j++) {
        if (col[j] < 0) continue;
        lump[row[i]] += M[i][j];
      }
    }
    ierr = MatSetValues(ctx->MatVVPC, d*N, row, d*N, col, &A[0][0], ADD_VALUES);CHKERRQ(ierr);
    if (ctx->options->debug > 1) {
      ierr = MatAssemblyBegin(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatView(ctx->MatVVPC, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(ctx->massLump, &lump);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->deta, &deta); CHKERRQ(ierr);
  ierr = VecRestoreArrays(ctx->strain, d, &strain); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->coord, &x); CHKERRQ(ierr);
  ierr = ISRestoreIndices(ctx->isLV, &ixL); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->MatVVPC, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecReciprocal(ctx->massLump);CHKERRQ(ierr);
  ierr = MatDiagonalScale(ctx->MatVVPC, ctx->massLump, PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPVelocity, ctx->MatVV, ctx->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPSchurVelocity, ctx->MatVV, ctx->MatVVPC, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->KSPSchur, ctx->MatSchur, ctx->MatSchur, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  //ierr = MatView(ctx->MatVVPC, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
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
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(c->MatPV, c->vG1, c->pG0);CHKERRQ(ierr);
  ierr = VecScale(c->pG0, -1.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGP, x, c->pG0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGP, x, c->pG0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // KSPSolve tampers with vG0 and vG1
  ierr = KSPSolve(c->KSPSchur, c->pG0, c->pG1);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterPG, c->pG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(c->MatVP, c->pG1, c->vG0);CHKERRQ(ierr);
  ierr = VecScale(c->vG0, -1.0);CHKERRQ(ierr);
  ierr = KSPSolve(c->KSPVelocity, c->vG0, c->vG1);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterVG, c->vG1, y, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterPG, c->pG1, y, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterVG, c->vG1, y, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesStateView"
PetscErrorCode StokesStateView(StokesCtx *ctx, Vec state, const char *filename)
{
  PetscViewer    view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(ctx->scatterGP, state, ctx->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGP, state, ctx->pG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGV, state, ctx->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGV, state, ctx->vG0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterPL, ctx->pG0, ctx->workP[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPL, ctx->pG0, ctx->workP[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, ctx->vG0, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, ctx->vG0, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesMixedVelocity(ctx, ctx->workV[0]);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(ctx->workP[0], ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterDL, ctx->dirichlet, ctx->workV[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(ctx->scatterGP, ctx->force, ctx->pG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGP, ctx->force, ctx->pG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterGV, ctx->force, ctx->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterGV, ctx->force, ctx->vG1, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterPL, ctx->pG1, ctx->workP[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterPL, ctx->pG1, ctx->workP[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterVL, ctx->vG1, ctx->workV[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterVL, ctx->vG1, ctx->workV[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = StokesMixedVelocity(ctx, ctx->workV[1]);CHKERRQ(ierr);
  ierr = StokesPressureReduceOrder(ctx->workP[1], ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterDL, ctx->dirichlet, ctx->workV[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterDL, ctx->dirichlet, ctx->workV[1], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);


  ierr = PetscViewerCreate(PETSC_COMM_SELF, &view);CHKERRQ(ierr);
  ierr = PetscViewerSetType(view, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  //ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);  // Maybe there is a way to make this work properly?
  ierr = PetscViewerFileSetName(view, "stokes.vtk");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(view, "# vtk DataFile Version 2.0\nStokes Output\nASCII\nDATASET STRUCTURED_GRID\n");CHKERRQ(ierr);
  {
    PetscInt d = ctx->numDims, *dim = ctx->dim, m = dim[0], n = dim[1], p = (d > 2) ? dim[2] : 1;
    PetscInt nodes = productInt(d, dim);
    ierr = PetscViewerASCIIPrintf(view, "DIMENSIONS %d %d %d\nPOINTS %d double\n", m, n, p, m*n*p);CHKERRQ(ierr);
    ierr = StokesVecView(ctx->coord, nodes, d, 3, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nPOINT_DATA %d\nVECTORS velocity double\n", m*n*p);CHKERRQ(ierr);
    ierr = StokesVecView(ctx->workV[0], nodes, d, 3, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nSCALARS pressure double 1\nLOOKUP_TABLE default\n");CHKERRQ(ierr);
    ierr = StokesVecView(ctx->workP[0], nodes, 1, 1, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nVECTORS vel_force double\n", m*n*p);CHKERRQ(ierr);
    ierr = StokesVecView(ctx->workV[1], nodes, d, 3, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nSCALARS div_force double 1\nLOOKUP_TABLE default\n");CHKERRQ(ierr);
    ierr = StokesVecView(ctx->workP[1], nodes, 1, 1, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nSCALARS eta double 1\nLOOKUP_TABLE default\n");CHKERRQ(ierr);
    ierr = StokesVecView(ctx->eta, nodes, 1, 1, view);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view, "\nSCALARS deta double 1\nLOOKUP_TABLE default\n");CHKERRQ(ierr);
    ierr = StokesVecView(ctx->deta, nodes, 1, 1, view);CHKERRQ(ierr);
    {
      PetscReal **strain;
      ierr = PetscViewerASCIIPrintf(view, "\nTENSORS strain double\n");CHKERRQ(ierr);
      ierr = VecGetArrays(ctx->strain, d, &strain);CHKERRQ(ierr);
      for (int i=0; i < nodes; i++) {
        for (int j=0; j < 3; j++) {
          for (int k=0; k < 3; k++) {
            ierr = PetscViewerASCIIPrintf(view, "%20e ", (j<d && k<d) ? strain[j][i*d+k] : 0.0);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(view, "\n");
        }
        ierr = PetscViewerASCIIPrintf(view, "\n");
      }
      ierr = VecRestoreArrays(ctx->strain, d, &strain);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerDestroy(view);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesVecView"
PetscErrorCode StokesVecView(Vec v, PetscInt nodes, PetscInt pernode, PetscInt perline, PetscViewer view)
{
  PetscReal     *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for (int i=0; i < nodes; i++) {
    for (int j=0; j < pernode && j < perline; j++) {
      ierr = PetscViewerASCIIPrintf(view, "%20e ", x[i*pernode+j]);CHKERRQ(ierr);
    }
    for (int j=pernode; j < perline; j++) {
      ierr = PetscViewerASCIIPrintf(view, "0 ");
    }
    ierr = PetscViewerASCIIPrintf(view, "\n");
  }
  ierr = VecRestoreArray(v, &x); CHKERRQ(ierr);
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
#define __FUNCT__ "StokesRheologyPower"
PetscErrorCode StokesRheologyPower(PetscInt d, PetscReal gamma, PetscReal *eta, PetscReal *deta, void *ctx)
{
  StokesOptions *opt = (StokesOptions *)ctx;
  PetscReal      n = opt->exponent;
  PetscReal      p = (1.0 - n) / (2.0 * n);

  PetscFunctionBegin;
  *eta = opt->hardness * pow(opt->regularization + gamma / opt->gamma0, p);
  if (PetscAbs(n) > 1.0e-5) { // Avoid a singularity for the special case
    *deta = opt->hardness * p / opt->gamma0 * pow(opt->regularization + gamma / opt->gamma0, p - 1.0);
  } else {
    *deta = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesExact0"
PetscErrorCode StokesExact0(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
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
#define __FUNCT__ "StokesExact1"
PetscErrorCode StokesExact1(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{
  const PetscReal eta = 1.0;
  PetscReal u, v, p;

  PetscFunctionBegin; InFunction;
  if (d > 3) SETERRQ2(1, "%s only implemented for dimension 2 and 3 but %d given", __FUNCT__, d);
  u =  sin(0.5 * PETSC_PI * coord[0]) * cos(0.5 * PETSC_PI * coord[1]);
  v = -cos(0.5 * PETSC_PI * coord[0]) * sin(0.5 * PETSC_PI * coord[1]);
  p = 0.25 * (cos(PETSC_PI * coord[0]) + cos(PETSC_PI * coord[1])) + 10 * (coord[0] + coord[1]);
  if (value) {
    value[0] = u;
    value[1] = v;
    if (d == 3) value[2] = 0.0;
    value[d] = p;
  }
  if (rhs) {
    rhs[0]   = PetscSqr(0.5 * PETSC_PI) * eta * u - 0.25 * PETSC_PI * sin(PETSC_PI * coord[0]) + 10;
    rhs[1]   = PetscSqr(0.5 * PETSC_PI) * eta * v - 0.25 * PETSC_PI * sin(PETSC_PI * coord[1]) + 10;
    if (d == 3) rhs[2] = 0.0;
    rhs[d]   = 0.0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "StokesExact2"
PetscErrorCode StokesExact2(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{
  const PetscReal eta = 1.0;
  PetscReal u, v, p;

  PetscFunctionBegin; InFunction;
  if (d > 3) SETERRQ2(1, "%s only implemented for dimension 2 but %d given", __FUNCT__, d);
  u =  sin(0.5 * PETSC_PI * coord[0]) * cos(0.5 * PETSC_PI * coord[1]);
  v = -cos(0.5 * PETSC_PI * coord[0]) * sin(0.5 * PETSC_PI * coord[1]);
  p = 0.0;
  if (value) {
    value[0] = u; value[1] = v; value[2] = p;
    if (d == 3) value[2] = 0.0;
  }
  if (rhs) {
    rhs[0]   = PetscSqr(0.5 * PETSC_PI) * eta * u;
    rhs[1]   = PetscSqr(0.5 * PETSC_PI) * eta * v;
    if (d == 3) rhs[2] = 0.0;
    rhs[d]   = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesExact3"
PetscErrorCode StokesExact3(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{
  const PetscReal eta = 1.0;
  PetscReal u, v, p;

  PetscFunctionBegin; InFunction;
  if (d != 2) SETERRQ2(1, "%s only implemented for dimension 2 but %d given", __FUNCT__, d);
  u = coord[1] + 1.0;
  v = 0.0;
  p = 0.0;
  if (value) {
    value[0] = u; value[1] = v; value[2] = p;
  }
  if (rhs) {
    rhs[0]   = 0.0;
    rhs[1]   = 0.0;
    rhs[2]   = 0.0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "StokesDirichlet"
PetscErrorCode StokesDirichlet(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *void_ctx)
{
  StokesExactBoundaryCtx *ctx = (StokesExactBoundaryCtx *)void_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin; InFunction;
  *type = DIRICHLET;
  ierr = ctx->exact(d, coord, value, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesBoundary1"
PetscErrorCode StokesBoundary1(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *void_ctx)
{
  const PetscReal         epsilon = 1e-7;
  StokesExactBoundaryCtx *ctx     = (StokesExactBoundaryCtx *)void_ctx;
  bool                    inside;
  PetscErrorCode          ierr;

  PetscFunctionBegin; InFunction;
  inside = false;
  for (int i=0; i < d-1; i++) inside |= (PetscAbs(coord[i]) < 0.999);
  if (coord[d-1] > 0.999 && inside) { // Impose condition at the 'surface'
    PetscReal x[d], v[d], w[d], vel[d][d];
    *type = NEUMANN;
    for (int i=0; i < d; i++) {
      ierr = ctx->exact(d, coord, v, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
      for (int j=0; j < d; j++) { x[j] = coord[j] + epsilon * ((i == j) ? 1 : 0); }
      ierr = ctx->exact(d, x, w, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
      if (false) { // first order
        for (int j=0; j < d; j++) { vel[j][i] = (1.0 / epsilon) * (w[j] - v[j]); }
      } else { // centered difference
        for (int j=0; j < d; j++) { vel[j][i] = (0.5 / epsilon) * w[j]; }
        for (int j=0; j < d; j++) { x[j] = coord[j] - epsilon * ((i == j) ? 1 : 0); }
        ierr = ctx->exact(d, x, w, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
        for (int j=0; j < d; j++) { vel[j][i] -= (0.5 / epsilon) * w[j]; }
      }
    }
    for (int i=0; i < d; i++) { // multiply velocity gradient tensor with normal vector
      value[i] = 0.0;
      for (int j=0; j < d; j++) {
        value[i] += 0.5 * (vel[j][i] + vel[i][j]) * normal[j];
      }
    }
  } else { // Simply impose dirichlet conditions
    *type = DIRICHLET;
    ierr = ctx->exact(d, coord, value, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesBoundary2"
PetscErrorCode StokesBoundary2(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *void_ctx)
{
  const PetscReal         epsilon = 1e-7;
  StokesExactBoundaryCtx *ctx     = (StokesExactBoundaryCtx *)void_ctx;
  bool                    inside;
  PetscErrorCode          ierr;

  PetscFunctionBegin; InFunction;
  inside = false;
  for (int i=0; i < d-1; i++) inside |= (PetscAbs(coord[i]) < 0.999);
  if (coord[d-1] > 0.999 && inside) { // Impose condition at the 'surface'
    PetscReal x[d], v[d], w[d], vel[d][d];
    *type = NEUMANN;
    for (int i=0; i < d; i++) {
      ierr = ctx->exact(d, coord, v, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
      for (int j=0; j < d; j++) { x[j] = coord[j] + epsilon * ((i == j) ? 1 : 0); }
      ierr = ctx->exact(d, x, w, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
      if (false) { // first order
        for (int j=0; j < d; j++) { vel[j][i] = (1.0 / epsilon) * (w[j] - v[j]); }
      } else { // centered difference
        for (int j=0; j < d; j++) { vel[j][i] = (0.5 / epsilon) * w[j]; }
        for (int j=0; j < d; j++) { x[j] = coord[j] - epsilon * ((i == j) ? 1 : 0); }
        ierr = ctx->exact(d, x, w, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
        for (int j=0; j < d; j++) { vel[j][i] -= (0.5 / epsilon) * w[j]; }
      }
    }
    for (int i=0; i < d; i++) { // multiply velocity gradient tensor with normal vector
      value[i] = 0.0;
      for (int j=0; j < d; j++) {
        value[i] += 0.5 * (vel[j][i] + vel[i][j]) * normal[j];
      }
    }
  } else if (coord[d-1] < -0.999) { // Impose mixed condition at the bed
    *type = MIXED;
    value[0] = 1.0; // alpha
    for (int i=0; i < d; i++) { value[1+i] = 0.0; } // Zero flux through boundary
  } else { // Simply impose dirichlet conditions
    *type = DIRICHLET;
    ierr = ctx->exact(d, coord, value, PETSC_NULL, ctx->exactCtx);CHKERRQ(ierr);
  }
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
