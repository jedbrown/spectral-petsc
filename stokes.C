static char help[] = "Stokes problem with non-Newtonian rheology via Chebyshev collocation.\n";

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
  Options    *options;
  PetscInt    numFields, numDims, numWork;
  PetscInt   *dims;
  Mat        *Dfield;           // derivative matrices, operates on vector valued local Vec
  Vec        *work, *gradu;     // each is vector valued
  Vec         coord;            // vector valued
  Vec         eta;              // effective viscosity, scalar valued
  Vec         deta;             // vector valued, derivative of eta with respect to field components (velocity, pressure)
  Vec         dirichlet;        // special dirichlet format
  Vec         force;            // Global body force vector
  IS          isLG, isGL;       // vector valued local <-> global
  VecScatter  scatterLG, scatterGL;
  IS          isDL;
  VecScatter  scatterLD, scatterDL; // vector valued local <-> special dirichlet
} StokesCtx;

PetscErrorCode StokesCreate(MPI_Comm comm, Mat *A, Vec *x, StokesCtx **);
PetscErrorCode StokesDestroy(StokesCtx *);
PetscErrorCode StokesProcessOptions(MPI_Comm, StokesCtx *);
PetscErrorCode StokesMatMult(Mat, Vec, Vec);
PetscErrorCode StokesFunction(SNES, Vec, Vec, void *);
PetscErrorCode StokesJacobian(SNES, Vec, Mat *, Mat *, MatStructure *, void *);
PetscErrorCode StokesSetupDomain(StokesCtx *);
PetscErrorCode StokesCreateExactSolution(SNES, Vec u, Vec u2);
PetscErrorCode StokesCheckResidual(SNES snes, Vec u, Vec x);
PetscErrorCode StokesRemoveConstantPressure(KSP, StokesCtx *, Vec *, MatNullSpace *);
//PetscErrorCode StokesRheologyPowerLaw(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta);
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta, void *ctx);
PetscErrorCode StokesExactNull(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx);
PetscErrorCode StokesDirichletZero(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx);

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
  Vec                  x, r, b, b0, u, u2, nv;
  PetscReal            norm, rnorm, unorm, u2norm;
  PetscInt             its, m;
  SNESConvergedReason  reason;
  PetscTruth           flag;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);
  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);
  ierr = StokesCreate(comm, &A, &x, &ctx);CHKERRQ(ierr);
  ierr = VecGetSize(x, &m);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(comm, m, m, 1+2*ctx->numDims, PETSC_NULL, &P); CHKERRQ(ierr);
  ierr = VecDuplicate(x, u);CHKERRQ(ierr);
  ierr = VecDuplicate(x, u2);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes, ctx);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, StokesFunction, ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, P, StokesJacobian, ctx);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPFGMRES);CHKERRQ(ierr);
  ierr = StokesRemoveConstantPressure(ksp, ctx, &nv, &ns);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCILU);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc, 2);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // u = exact solution, u2 = A(u) u (used as forcing term)
  ierr = CreateExactSolution(snes, u, u2);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, &unorm);CHKERRQ(ierr);
  ierr = VecNorm(u2, NORM_INFINITY, &u2norm);CHKERRQ(ierr);

  ierr = VecSet(x, 0.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes, PETSC_NULL, x);CHKERRQ(ierr);

  ierr = VecAXPY(x, -1.0, u); CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(x, x, u);CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &rnorm);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of nonlinear iterations = %d\n", its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%-25s: abs = %8e   rel = %8e\n", "Norm of error", norm, rnorm);CHKERRQ(ierr);

  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = StokesDestroy(ctx);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);           ierr = VecDestroy(u2);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(ns);CHKERRQ(ierr); ierr = VecDestroy(nv);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesCreate"
PetscErrorCode StokesCreate(MPI_Comm comm, Mat *A, Vec *X, StokesCtx **ctx)
{
  StokesCtx *c;
  Options *opt;
  PetscInt d, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(1, StokesCtx, ctx, 1, StokesOptions, &opt, 1);CHKERRQ(ierr);
  c = *ctx; c->options = opt;
  ierr = StokesProcessOptions(c);CHKERRQ(ierr);
  d = c->numDims;
  ierr = PetscMalloc(d*sizeof(Mat), &c->Dfield);CHKERRQ(ierr);
  m = productInt(d, dim); // number of local nodes
  c->numWork = 2*d+1; // very generous
  ierr = VecCreate(comm, &c->eta);CHKERRQ(ierr); // Scalar valued
  ierr = VecSetSizes(c->eta, m, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecCreate(comm, &c->coord);CHKERRQ(ierr); // Vector valued, one component for each dimension
  ierr = VecSetSizes(c->coord, m*d, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->coord, d);CHKERRQ(ierr);
  ierr = VecCreate(comm, &c->deta);CHKERRQ(ierr); // Vector valued, one component for each dimension plus pressure
  ierr = VecSetSizes(c->deta, m*(d+1), PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->deta, d+1);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->deta, c->numWork, &c->work); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->deta, d, &c->gradu); CHKERRQ(ierr);
  { // Create differentiation matrices
    PetscInt cheb_dim[d+1];
    for (int i=0; i < d; i++) cheb_dim[i] = ctx->dims[i];
    cheb_dim[d] = d+1;
    for (int i=0; i < d; i++) {
      ierr = MatCreateCheb(comm, d+1, i, cheb_dim, fftw_flag, c->work[0], c->work[1], &c->Dfield[i]);CHKERRQ(ierr);
    }
  }
  { // Setup coordinate mapping
    PetscReal *x;
    ierr = VecGetArray(c->coord, &x);CHKERRQ(ierr);
    for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
      for (int j=0; j < d; j++) {
        x[it.i*d+j] = cos(it.ind[j] * PETSC_PI / (dim[j] - 1));
      }
    }
    ierr = VecRestoreArray(c->x, &x);CHKERRQ(ierr);
  }
  ierr = StokesSetupDomain(c);CHKERRQ(ierr);
  {
    PetscInt n;
    ierr = VecDuplicateVec(c->force, X);CHKERRQ(ierr);
    ierr = VecGetSize(*X, &n);CHKERRQ(ierr);
    ierr = MatCreateShell(comm, n, n, PETSC_DECIDE, PETSC_DECIDE, c, A);CHKERRQ(ierr);
  }
  ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))MatMult_Stokes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesDestroy"
PetscErrorCode StokesDestroy (StokesCtx *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (int i=0; i < c->numDims; i++) {
    ierr = MatDestroy(c->Dfield[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(c->work, c->numWork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->gradu, c->d);CHKERRQ(ierr);
  ierr = VecDestroy(c->coord);CHKERRQ(ierr);
  ierr = VecDestroy(c->eta);CHKERRQ(ierr);
  ierr = VecDestroy(c->deta);CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet);CHKERRQ(ierr);
  ierr = ISDestroy(c->isLG);CHKERRQ(ierr);
  ierr = ISDestroy(c->isGL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterGL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLG);CHKERRQ(ierr);
  ierr = ISDestroy(c->isDL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterDL);CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLD);CHKERRQ(ierr);
  ierr = PetscFree(c->Dfield);CHKERRQ(ierr);
  ierr = PetscFree(c->dims);CHKERRQ(ierr);
  ierr = PetscFree2(c, c->options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNC__
PetscErrorCode StokesProcessOptions(MPI_Comm comm, StokesCtx *ctx)
{
  Options        *opt = ctx->options;
  PetscTruth      flag;
  PetscInt        exact, rheology;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ctx->numDims = 10;
  ierr = PetscMalloc(ctx->numDims*sizeof(PetscInt), &ctx->dims);CHKERRQ(ierr);
  exact = 0; rheology = 0;
  opt->debug = 0; opt->softness = 1.0; opt->exponent = 1.0; opt->regularization = 1.0;
  ierr = PetscOptionsBegin(comm, "", "Stokes problem options", "");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dims", "list of dimension extent", "stokes.C", ctx->dims, &ctx->numDims, &flag);CHKERRQ(ierr);
  if (!flag) { ac->dims = 2; ctx->dims[0] = 8; ctx->dims[1] = 6; }
  ierr = PetscOptionsInt("-debug", "debugging level", "stokes.C", ctx->debug, &ctx->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-exact", "exact solution type", "stokes.C", exact, &exact, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rheology", "rheology type", "stokes.C", rheology, &rheology, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hardness", "power law hardness parameter", "stokes.C", opt->hardness, &opt->hardness, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-exponent", "power law exponent", "stokes.C", opt->exponent, &opt->exponent, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-regularization", "regularization parameter for viscosity", "stokes.C", opt->regularization, &opt->regularization, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscPrintf(comm, "Stokes problem");CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  dims = [", ctx->numDims);CHKERRQ(ierr);
  for (int i=0; i < ctx->numDims; i++) {
    if (i > 0) { ierr = PetscPrintf(comm, ",");CHKERRQ(ierr); }
    ierr = PetscPrintf(comm, "%d", ctx->dims[i]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm, "]\n  hardness = %f    exponent = %8f    regularization = %8y\n", ctx->hardness, ctx->exponent, ctx->regularization);CHKERRQ(ierr);

  switch (exact) {
    case 0:
      opt->exact       = StokesExactNull;
      opt->exactCtx    = PETSC_NULL;
      opt->boundary    = StokesBdyDirichletZero;
      opt->boundaryCtx = PETSC_NULL;
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP, "Exact solution %d not implemented", exact);
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
  d = c->numDims; F = d+1; n = productInt(d, c->dims);
  xL = c->work[0]; U = &c->work[1]; V = &c->work[d+1];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // FIXME: project interior pressure to boundary nodes

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
      x[i*F+j] -= u[j][i*F+d]; // pressure term
      x[i*F+d] += u[j][i*F+j]; // divergence of velocity
      for (int k=0; k < d; k++) {
        x[i*F+j] += v[k][i*F+j]; // divergence of stress
      }
    }
  }
  ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
  ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);

  ierr = VecScatterBegin(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesFunction"
PetscErrorCode StokesFunction(SNES snes, Vec uG, Vec rhs, void *ctx)
{
  StokesCtx *c = (StokesCtx *)ctx;
  PetscInt n, d, F;
  Vec xL, *V, *W;
  PetscReal **u, **v, *eta, *deta, *x, *stretching;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  d = c->numDims; F = d+1; n = productInt(d, c->dims);
  xL = c->work[0]; U = &c->work[1]; V = &c->work[d+1];
  ierr = VecZeroEntries(xL);CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, xG, xL, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  // FIXME: project interior pressure to boundary nodes

  for (int i=0; i < d; i++) { ierr = MatMult(c->Dfield[i], xL, gradu[i]);CHKERRQ(ierr); }

  { // In `U' we have gradients [u_x v_x w_x p_x] [u_y v_y w_y p_y] [u_z v_z w_z p_z]
    PetscReal stretching[d*d];
    ierr = VecGetArrays(c->gradu, d, &u);CHKERRQ(ierr);
    ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
    ierr = VecGetArray(c->eta, &eta);CHKERRQ(ierr);
    ierr = VecGetArray(c->deta, &deta);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      for (int j=0; j < n; j++) {
        for (int k=0; k < n; k++) {
          stretching[j][k] = 0.5 * (u[j][i*F+k] + u[k][i*F+j]);
        }
      }
      ierr = c->options->rheology(d, stretching, &eta[i], &deta[i]);CHKERRQ(ierr);
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

  { // In `V' we have [(e u_x)_x (e v_x)_x (e w_x)_x p_xx] [(e u_y)_y (e v_y)_y (e w_y)_y p_yy] [...]
    ierr = VecZeroEntries(xL);CHKERRQ(ierr);
    ierr = VecGetArray(xL, &x);CHKERRQ(ierr);
    ierr = VecGetArrays(U, d, &u);CHKERRQ(ierr);
    ierr = VecGetArrays(V, d, &v);CHKERRQ(ierr);
    for (int i=0; i < n; i++) { // each node
      for (int j=0; j < d; j++) {
        x[i*F+j] -= u[j][i*F+d]; // pressure term
        x[i*F+d] += u[j][i*F+j]; // divergence of velocity
        for (int k=0; k < d; k++) {
          x[i*F+j] += v[k][i*F+j]; // divergence of stress
        }
      }
    }
    ierr = VecRestoreArray(xL, &x);CHKERRQ(ierr);
    ierr = VecRestoreArrays(U, d, &u);CHKERRQ(ierr);
    ierr = VecRestoreArrays(V, d, &v);CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, xL, yG, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecAXPY(yG, -1.0, c->force);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesJacobian"
PetscErrorCode StokesJacobian(SNES snes, Vec w, Mat *A, Mat *P, MatStructure *flag, void *ctx)
{

  PetscFunctionBegin;
  // The nonlinear term has already been fixed up by StokesFunction() so we just need to deal with the preconditioner here.
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "StokesSetupDomain"
PetscErrorCode StokesSetupDomain(StokesCtx *c)
{
  const PetscInt d = c->numDims, *dim = c->dims, m = productInt(d, dim);;
  IS isG, isD;
  PetscInt *ixL, *ixG, *ixD, *ind, m, l, g ,b;
  PetscReal *uD, *x, *n, nn;
  BdyType bdy;
  Vec vL, vG, vD;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc5(m, PetscInt, &ixL, m, PetscInt, &ixG, m, PetscInt, &ixD, d, PetscInt, &ind, d, PetscReal, &n);CHKERRQ(ierr);
  ierr = VecGetArray(c->work[0], &uD); CHKERRQ(ierr); // Some workspace for boundary values
  ierr = VecGetArray(c->coord, &x); CHKERRQ(ierr);    // Coordinates in a block-size d vector
  l = 0; g = 0; b = 0; // indices for local, global, and dirichlet boundary
  for (BlockIt it = BlockIt(c->d, c->dim); !it.done; it.next()) {
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
          ixLG[l] = -1;
          ixDL[b++] = l++;
        }
      } else { SETERRQ(1, "Neumann not implemented."); }
    } else { // Interior
      for (int k=0; k < d+1; k++) { // same mapping for each field
        ixL[l] = g;
        ixG[g++] = l++;
      }
    }
  }
  ierr = VecRestoreArray(c->coord, &x); CHKERRQ(ierr);    // Coordinates in a block-size d vector

  ierr = VecCreate(comm, &vG);CHKERRQ(ierr);              // Prototype global vector
  ierr = VecSetSizes(vG, g, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecCreate(comm, &vD);CHKERRQ(ierr);              // Special dirichlet vector
  ierr = VecSetSizes(vD, b, PETSC_DECIDE); CHKERRQ(ierr);
  { // Fill dirichlet values into the special dirichlet vector.
    PetscScalar *v;
    ierr = VecGetArray(vD, &v); CHKERRQ(ierr);
    for (int i=0; i < b; i++) v[i] = uD[i];
    ierr = VecRestoreArray(vD, &v); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->work[0], &uD); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "DOF distribution: %8d local     %8d global     %8d dirichlet\n", l, g, b); CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm, l, ixL, &c->isLG); CHKERRQ(ierr); // We need this to build the preconditioner
  ierr = ISCreateGeneral(comm, d, ixD, &c->isDL); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, g, ixG, &c->isGL); CHKERRQ(ierr);

  vL = c->work[0]; // A prototype local vector
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, c->isDL, &c->scatterDL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, c->isGL, &c->scatterGL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isDL, vD, PETSC_NULL, &c->scatterLD); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, c->isGL, vG, PETSC_NULL, &c->scatterLG); CHKERRQ(ierr);

  ierr = PetscFree5(ixL, ixG, ixD, ind, n); CHKERRQ(ierr);
  c->force = vG;
  c->dirichlet = vD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesCreateExactSolution"
PetscErrorCode StokesCreateExactSolution(SNES snes, Vec U, Vec U2, Vec b)
{
  StokesCtx      *c;
  PetscInt        d;
  PetscScalar    *u, *v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes, (void **)&c); CHKERRQ(ierr);
  d = c->numDims; dim = c->dims;
  ierr = VecGetArray(c->work[0], &u); CHKERRQ(ierr);
  ierr = VecGetArray(c->work[1], &v); CHKERRQ(ierr);
  ierr = VecGetArray(c->coord, &coord); CHKERRQ(ierr);
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    const PetscInt i = it.i;
    ierr = c->options->exact(d, &coord[i*d], &u[i*(d+1)], &v[i*(d+1)], c->options->exactCtx);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->work[0], &u); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->work[1], &v); CHKERRQ(ierr);

  ierr = VecScatterBegin(c->scatterLG, c->work[0], U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->work[0], U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->work[1], U2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->work[1], U2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLD, c->work[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLD, c->work[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  ierr = VecSet(b, 0.0); CHKERRQ(ierr); // b is the inhomogenous dirichlet part, zero at interior nodes
  ierr = FormFunction(snes, b, c->force, (void *)c); CHKERRQ(ierr);
  ierr = VecAYPX(b, -1.0, b0); CHKERRQ(ierr); // Lift inhomogenous part to right hand side

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRemoveConstantPressure"
PetscErrorCode StokesRemoveConstantPressure(KSP ksp, StokesCtx *ctx, Vec *X, MatNullSpace *ns)
{
  MPI_Comm       comm;
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicateVec(ctx->force, X);CHKERRQ(ierr);
  ierr = VecGetSize(*X, &m);CHKERRQ(ierr);
  ierr = VecGetArray(*X, &x);
  for (int i=0; i < m; i++) {
    for (int j=0; j < ctx->numDims; j++) {
      x[i*(ctx->numDims+1)+j] = 0.0;
    }
    x[i*(ctx->numDims+1)+ctx->numDims] = 1.0;
  }
  ierr = VecRestoreArray(*X, &x);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ksp, &comm);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, 1, X, ns);CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ksp, *ns);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesRheologyLinear"
PetscErrorCode StokesRheologyLinear(PetscInt d, PetscReal *stretching, PetscReal *eta, PetscReal *deta, void *ctx)
{

  PetscFunctionBegin;
  *eta = 1.0;
  for (int i=0; i < d; i++) deta[i] = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesExactNull"
PetscErrorCode StokesExactNull(PetscInt d, PetscReal *coord, PetscReal *value, PetscReal *rhs, void *ctx)
{

  PetscFunctionBegin;
  for (int i=0; i < d; i++) {
    value[i] = 0.0;
    rhs[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesDirichletZero"
PetscErrorCode StokesDirichletZero(PetscInt d, PetscReal *coord, PetscReal *normal, BdyType *type, PetscReal *value, void *ctx)
{

  PetscFunctionBegin;
  *type = DIRICHLET;
  for (int i=0; i < d+1; i++) {
    value[i] = 0.0;
  }
  PetscFunctionReturn;
}
