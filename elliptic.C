static char help[] = "A nonlinear elliptic equation by Chebyshev differentiation.\n";

#define SOLVE 1
#define CHECK_EXACT 1

#include "chebyshev.h"
#include <petscsnes.h>
#include <stdbool.h>

class BlockIt {
  public:
  BlockIt(int d, int *dim) : d(d) {
    int s=1;
    this->dim = new int[d]; stride = new int[d]; ind = new int[d];
    for (int j=d-1; j>=0; j--) {
      this->dim[j] = dim[j];
      ind[j] = 0;
      stride[j] = s;
      s *= dim[j];
    }
    i = 0;
    done = s == 0;
  }
  ~BlockIt() {
    delete [] dim;
    delete [] stride;
    delete [] ind;
  }
  void next() {
    int carry = 1;
    i = 0;
    for (int j = d-1; j >= 0; j--) {
      ind[j] += carry;
      carry = 0;
      if (ind[j] == dim[j]) {
        ind[j] = 0;
        carry = 1;
      }
      i += ind[j] * stride[j];
    }
    done = (bool)carry;
  }
  int shift(int j, int s) const {
    const int is = ind[j] + s;
    if (is < 0 || is >= dim[j]) return -1;
    return i + s * stride[j];
  }
  bool done;
  int i, *ind;
  private:
  int d, *dim, *stride;
};

PetscScalar dotScalar(PetscInt d, PetscScalar x[], PetscScalar y[]) {
  PetscScalar dot = 0.0;
  for (PetscInt i=0; i<d; i++) {
    dot += x[i] * y[i];
  }
  return dot;
}

int sumInt(int d, int dim[]) {
  int z = 0;
  for (int i=0; i<d; i++) z += dim[i];
  return z;
}

int productInt(int d, int dim[]) {
  int z = 1;
  for (int i=0; i<d; i++) z *= dim[i];
  return z;
}

void zeroInt(int d, int v[]) {
  for (int i=0; i < d; i++) v[i] = 0;
}

typedef struct {
  int d, *dim, nw;
  Mat *D;
  Vec *w, *gradu;
  Vec eta, deta, x;
  Vec dirichlet, dirichlet0;
  IS isG, isL;
  VecScatter scatterLG, scatterGL, scatterLD, scatterDL;
} MatElliptic;

typedef struct {
  PetscInt exact, d, *dim;
  Mat A;
  Vec b;
  PetscReal gamma, exponent;
  int debug;
} AppCtx;

typedef enum { BDY_DIRICHLET, BDY_NEUMANN } BdyType;

typedef struct {
  BdyType type;
  PetscScalar value;
} BdyCond;

typedef PetscErrorCode(*BdyFunc)(int, double *, double *, BdyCond *);

PetscErrorCode MatCreate_Elliptic(MPI_Comm comm, int d, int *dim, unsigned flag, BdyFunc bf, Vec *vG, Mat *A);
PetscErrorCode MatMult_Elliptic(Mat, Vec, Vec);
PetscErrorCode MatDestroy_Elliptic(Mat);
PetscErrorCode SetupBC(MPI_Comm comm, BdyFunc bf, Vec *vGlob, MatElliptic *c);
PetscErrorCode DirichletBdy(int d, double *x, double *n, BdyCond *bc);
PetscErrorCode CreateExactSolution(SNES snes, Vec u, Vec u2);
PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
PetscErrorCode FormJacobian(SNES, Vec, Mat *, Mat *, MatStructure *, void *);
PetscErrorCode VecPrint2(Vec, PetscInt, PetscInt, const char *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  MPI_Comm       comm;
  Vec            x, r, u, u2;
  Mat            A, P;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  PetscReal      norm, unorm, u2norm, rnorm;
  PetscInt       its;
  SNESConvergedReason reason;
  PetscErrorCode ierr;
  AppCtx         *ac;
  PetscTruth     flag;

  PetscFunctionBegin;
  //ierr = PetscMallocSetDumpLog(); CHKERRQ(ierr);
  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);
  fftw_import_system_wisdom();
  ierr = PetscMalloc(sizeof(AppCtx), &ac); CHKERRQ(ierr);
  ac->d = 10; // Maximum number of dimensions
  ierr = PetscMalloc(ac->d*sizeof(PetscInt), &ac->dim); CHKERRQ(ierr);

  ac->dim[0] = 8; ac->dim[1] = 6; // default dimension extent
  ac->debug = 0; ac->exact = 0; ac->gamma = 0.0; ac->exponent = 2.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elliptic problem options", ""); CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dims", "list of dimension extent", "elliptic.C", ac->dim, &ac->d, &flag); CHKERRQ(ierr);
  if (!flag) ac->d = 2;
  ierr = PetscOptionsInt("-debug", "debugging level", "elliptic.C", ac->debug, &ac->debug, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-exact", "exact solution type", "elliptic.C", ac->exact, &ac->exact, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-gamma", "strength of nonlinearity", "elliptic.C", ac->gamma, &ac->gamma, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-exponent", "exponent of nonlinearity", "elliptic.C", ac->exponent, &ac->exponent, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Elliptic problem");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "  dims = [", ac->d);CHKERRQ(ierr);
  for (int i=0; i < ac->d; i++) {
    if (i > 0) { ierr = PetscPrintf(PETSC_COMM_WORLD, ",");CHKERRQ(ierr); }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%d", ac->dim[i]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "]    gamma = %f    exponent = %8f\n", ac->gamma, ac->exponent);CHKERRQ(ierr);

  ierr = MatCreate_Elliptic(PETSC_COMM_WORLD, ac->d, ac->dim, FFTW_ESTIMATE, DirichletBdy, &u, &A); CHKERRQ(ierr);
  {
    PetscInt m,n;
    ierr = MatGetSize(A, &m, &n); CHKERRQ(ierr);
    // ierr = MatCreate(PETSC_COMM_SELF, &P); CHKERRQ(ierr);
    // ierr = MatSetSizes(P, m, n, m, n); CHKERRQ(ierr);
    // ierr = MatSetType(P, MATUMFPACK); CHKERRQ(ierr);
    // ierr = MatPreallocateInitialize(PETSC_COMM_SELF, m, n, &dnz, &onz); CHKERRQ(ierr);
    // ierr = MatSetFromOptions(P); CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 1+2*ac->d, PETSC_NULL, &P); CHKERRQ(ierr);
  }

  //ierr = VecPrint2(ac->x, m, n*2, "coordinates"); CHKERRQ(ierr);

  ierr = VecDuplicate(u, &u2);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &ac->b);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, P, FormJacobian, ac);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, FormFunction, ac);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes, ac);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPFGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCILU);CHKERRQ(ierr);
  ierr = PCFactorSetLevels(pc, 2);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ac->A = A;
  // u = exact solution, u2 = A(u) u
  ierr = CreateExactSolution(snes,u,u2);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, &unorm);CHKERRQ(ierr);
  ierr = VecNorm(u2, NORM_INFINITY, &u2norm);CHKERRQ(ierr);

#if CHECK_EXACT
  //ierr = VecSet(r, 0.0); CHKERRQ(ierr);
  //ierr = FormFunction(snes, r, x, ac); CHKERRQ(ierr);
  ierr = FormFunction(snes, u, r, ac); CHKERRQ(ierr);
  if (ac->debug >= 2) {
    PetscInt m = ac->dim[0], n = ac->dim[1];
    ierr = VecPrint2(u, m-2, n-2, "exact u"); CHKERRQ(ierr); printf("\n");
    ierr = VecPrint2(u2, m-2, n-2, "exact u2"); CHKERRQ(ierr); printf("\n");
    ierr = VecPrint2(r, m-2, n-2, "discrete residual"); CHKERRQ(ierr); printf("\n");
  }
  ierr = VecNorm(r, NORM_INFINITY, &norm);CHKERRQ(ierr);
  //ierr = VecPrint2(r, ac->dim[0]-2, ac->dim[1]-2, "discrete residual"); CHKERRQ(ierr); printf("\n");
  ierr = VecPointwiseDivide(r, r, u2);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_INFINITY, &rnorm);CHKERRQ(ierr);
  //ierr = VecPrint2(r, ac->dim[0]-2, ac->dim[1]-2, "discrete residual"); CHKERRQ(ierr); printf("\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%-25s: abs = %8e   rel = %8e\n", "Norm of exact residual",norm,rnorm);CHKERRQ(ierr);
#endif

#if SOLVE
  ierr = VecSet(x, 0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes, PETSC_NULL, x); CHKERRQ(ierr);

  //ierr = VecPrint2(u, m-2, n-2, "exact u"); CHKERRQ(ierr); printf("\n");
  //ierr = VecPrint2(x, m-2, n-2, "computed u"); CHKERRQ(ierr);

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
  /* ierr = SNESGetKSP(snes, ksp); CHKERRQ(ierr); */
  /* ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr); */
  /* ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %A iterations %D\n", norm, its); CHKERRQ(ierr); */
  //ierr = VecPrint2(b0, m-2, n-2, "b0"); CHKERRQ(ierr);
#endif

  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(u2);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(ac->b);CHKERRQ(ierr);
  ierr = PetscFree(ac->dim);CHKERRQ(ierr);
  ierr = PetscFree(ac);CHKERRQ(ierr);

  //ierr = PetscMallocDumpLog(PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Elliptic"
PetscErrorCode MatCreate_Elliptic(MPI_Comm comm, int d, int *dim, unsigned flag, BdyFunc bf, Vec *vG, Mat *A) {
  PetscErrorCode ierr;
  MatElliptic *c;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(MatElliptic), &c); CHKERRQ(ierr);
  ierr = PetscMalloc2(d, PetscInt, &c->dim, d, Mat, &c->D); CHKERRQ(ierr);
  c->d = d;
  for (int i=0; i<d; i++) { c->dim[i] = dim[i]; }
  PetscInt m = productInt(d, dim);
  c->nw = 2+d;

  ierr = VecCreateSeq(comm, m, &c->eta); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->eta, c->nw, &c->w); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(c->eta, c->d, &c->gradu); CHKERRQ(ierr);
  ierr = VecDuplicate(c->eta, &c->deta); CHKERRQ(ierr);
  ierr = VecSet(c->eta, 1.0); CHKERRQ(ierr);
  ierr = VecSet(c->deta, 0.0); CHKERRQ(ierr);

  for (int i=0; i<d; i++) {
    ierr = MatCreateCheb(comm, d, i, dim, flag, c->w[0], c->w[1], &c->D[i]); CHKERRQ(ierr);
  }

  PetscScalar *x;
  ierr = VecCreateSeq(comm, m*d, &c->x); CHKERRQ(ierr);
  ierr = VecSetBlockSize(c->x, d); CHKERRQ(ierr);
  ierr = VecGetArray(c->x, &x); CHKERRQ(ierr);
  for (BlockIt it = BlockIt(d, dim); !it.done; it.next()) {
    for (int j=0; j < d; j++) {
      x[it.i*d+j] = cos(it.ind[j] * PETSC_PI / (dim[j] - 1));
    }
  }
  ierr = VecRestoreArray(c->x, &x); CHKERRQ(ierr);

  ierr = SetupBC(comm, bf, vG, c); CHKERRQ(ierr);

  PetscInt n;
  ierr = VecGetSize(*vG, &n); CHKERRQ(ierr);
  ierr = MatCreateShell(comm, n, n, n, n, c, A); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))MatMult_Elliptic); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))MatDestroy_Elliptic); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Elliptic"
PetscErrorCode MatMult_Elliptic(Mat A, Vec U, Vec V) {
  PetscErrorCode ierr;
  MatElliptic *c;
  PetscInt n;
  PetscScalar *u, **u_, **u0_, *eta, *deta;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, U, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, U, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet0, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet0, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  for (int d=0; d < c->d; d++) {
    ierr = MatMult(c->D[d], c->w[0], c->w[1+d]); CHKERRQ(ierr);
  }

  ierr = VecGetSize(c->w[0], &n); CHKERRQ(ierr);
  ierr = VecGetArray(c->w[0], &u); CHKERRQ(ierr);
  ierr = VecGetArrays(&c->w[1], c->d, &u_); CHKERRQ(ierr);
  ierr = VecGetArrays(c->gradu, c->d, &u0_); CHKERRQ(ierr);
  ierr = VecGetArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(c->deta, &deta); CHKERRQ(ierr);
  for (int i=0; i < n; i++) {
    for (int d=0; d < c->d; d++) {
      u_[d][i] = eta[i] * u_[d][i] + deta[i] * u[i] * u0_[d][i];
    }
  }
  ierr = VecRestoreArray(c->w[0], &u); CHKERRQ(ierr);
  ierr = VecRestoreArrays(&c->w[1], c->d, &u_); CHKERRQ(ierr);
  ierr = VecRestoreArrays(c->gradu, c->d, &u0_); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->deta, &deta); CHKERRQ(ierr);

  ierr = VecZeroEntries(c->w[0]); CHKERRQ(ierr);
  for (int d=0; d < c->d; d++) {
    ierr = MatMult(c->D[d], c->w[1+d], c->w[1+c->d]); CHKERRQ(ierr);
    ierr = VecAXPY(c->w[0], -1.0, c->w[1+c->d]); CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(c->scatterLG, c->w[0], V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[0], V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Elliptic"
PetscErrorCode MatDestroy_Elliptic (Mat A) {
  PetscErrorCode ierr;
  MatElliptic *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  for (int d=0; d < c->d; d++) {
    ierr = MatDestroy(c->D[d]); CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(c->w, c->nw); CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->gradu, c->d); CHKERRQ(ierr);
  ierr = VecDestroy(c->x); CHKERRQ(ierr);
  ierr = VecDestroy(c->eta); CHKERRQ(ierr);
  ierr = VecDestroy(c->deta); CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet); CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet0); CHKERRQ(ierr);
  ierr = ISDestroy(c->isG); CHKERRQ(ierr);
  ierr = ISDestroy(c->isL); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterGL); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterDL); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLG); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLD); CHKERRQ(ierr);
  ierr = PetscFree2(c->dim, c->D); CHKERRQ(ierr);
  ierr = PetscFree(c);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupBC"
PetscErrorCode SetupBC(MPI_Comm comm, BdyFunc bf, Vec *vGlob, MatElliptic *c) {
  PetscErrorCode ierr;
  IS isG, isD;
  PetscInt *ixL, *ixG, *ixD, *ind, m, l, g ,d;
  PetscScalar *uD, *x, *n;
  Vec vL, vG, vD;
  //  const PetscInt m=c->dim[0], n=c->dim[1];

  PetscFunctionBegin;
  m = productInt(c->d, c->dim);
  ierr = PetscMalloc5(m, PetscInt, &ixL, m, PetscInt, &ixG, m, PetscInt, &ixD,
                      c->d, PetscInt, &ind, c->d, PetscScalar, &n); CHKERRQ(ierr);
  ierr = VecGetArray(c->w[1], &uD); CHKERRQ(ierr); // Just some workspace for boundary values
  ierr = VecGetArray(c->x, &x); CHKERRQ(ierr);    // Coordinates in a block-size d vector
  l = 0; g = 0; d = 0; // indices for local, global, and dirichlet
  for (BlockIt it = BlockIt(c->d, c->dim); !it.done; it.next()) {
    for (int j=0; j < c->d; j++) {
      if (it.ind[j] == 0) {
        n[j] = -1.0;
      } else if (it.ind[j] == c->dim[j] - 1) {
        n[j] = 1.0;
      } else {
        n[j] = 0.0;
      }
    }
    PetscScalar nn = dotScalar(c->d, n, n);
    if (nn > 1e-5) { // We are on the boundary
      BdyCond bc;
      for (int j=0; j < c->d; j++) n[j] /= sqrt(nn); // normalize n
      ierr = bf(c->d, &x[c->d*it.i], n, &bc); CHKERRQ(ierr);
      if (bc.type == BDY_DIRICHLET) {
          uD[d] = bc.value;
          ixL[l] = -1;
          ixD[d++] = l++;
        } else { SETERRQ(1, "Neumann not implemented."); }
    } else { // Interior
      ixL[l] = g;
      ixG[g++] = l++;
    }
  }
  ierr = VecRestoreArray(c->x, &x); CHKERRQ(ierr);    // Coordinates in a block-size d vector

  ierr = VecCreateSeq(comm, g, &vG); CHKERRQ(ierr);
  ierr = VecCreateSeq(comm, d, &vD); CHKERRQ(ierr);
  { // Fill dirichlet values into the special dirichlet vector.
    PetscScalar *v;
    ierr = VecGetArray(vD, &v); CHKERRQ(ierr);
    for (int i=0; i<d; i++) v[i] = uD[i];
    ierr = VecRestoreArray(vD, &v); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->w[1], &uD); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "DOF distribution: %8d local     %8d global     %8d dirichlet\n", l, g, d); CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm, l, ixL, &c->isL); CHKERRQ(ierr); // We need this to build the preconditioner
  ierr = ISCreateGeneral(comm, d, ixD, &isD); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, g, ixG, &isG); CHKERRQ(ierr);

  vL = c->w[0]; // A prototype local vector
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, isD, &c->scatterDL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, isG, &c->scatterGL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, isD, vD, PETSC_NULL, &c->scatterLD); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, isG, vG, PETSC_NULL, &c->scatterLG); CHKERRQ(ierr);

#define TEST_SCATTER 0
#if TEST_SCATTER
  printf("D -> L\n"); ierr = ISView(isD, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  //ierr = VecView(vD, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  printf("G -> L\n"); ierr = ISView(isG, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  //ierr = VecView(vG, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  //ierr = VecView(vL, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  ierr = VecSet(vL, 0.0); CHKERRQ(ierr);
  ierr = VecSet(vG, 1.0); CHKERRQ(ierr);
  ierr = VecSet(vD, 2.0); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, vG, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, vG, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, vD, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, vD, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  //ierr = VecView(vL, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  ierr = VecPrint2(vL, c->dim[0], c->dim[1], "local dof"); CHKERRQ(ierr); printf("\n");
  ierr = VecSet(vG, 0.0); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, vL, vG, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, vL, vG, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecPrint2(vG, c->dim[0]-2, c->dim[1]-2, "global dof"); CHKERRQ(ierr); printf("\n");
#endif

  ierr = PetscFree5(ixL, ixG, ixD, ind, n); CHKERRQ(ierr);
  c->isG = isG;
  ierr = ISDestroy(isD); CHKERRQ(ierr);
  *vGlob = vG;
  c->dirichlet = vD;
  ierr = VecDuplicate(vD, &c->dirichlet0); CHKERRQ(ierr);
  ierr = VecZeroEntries(c->dirichlet0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DirichletBdy"
PetscErrorCode DirichletBdy(int dim, double *x, double *n, BdyCond *bc) {
  //PetscErrorCode ierr;

  PetscFunctionBegin;
  bc->type = BDY_DIRICHLET;
  bc->value = 0.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes, Vec U, Vec rhs, void *void_ac) {
  PetscErrorCode ierr;
  AppCtx *ac = (AppCtx *)void_ac;
  MatElliptic *c;
  PetscInt n;
  PetscScalar *u, **u_, *eta, *deta, **w_;

  PetscFunctionBegin;
  ierr = MatShellGetContext(ac->A, (void **)&c); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, U, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, U, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet, c->w[0], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
#if DEBUG
  ierr = VecPrint2(c->w[0], c->dim[0], c->dim[1], "in function"); CHKERRQ(ierr); printf("\n");
#endif
  for (int d=0; d < c->d; d++) {
    ierr = MatMult(c->D[d], c->w[0], c->gradu[d]); CHKERRQ(ierr);
  }

  ierr = VecGetSize(c->w[0], &n); CHKERRQ(ierr);
  ierr = VecGetArray(c->w[0], &u); CHKERRQ(ierr);
  ierr = VecGetArrays(c->gradu, c->d, &u_); CHKERRQ(ierr);
  ierr = VecGetArrays(&c->w[1], c->d, &w_); CHKERRQ(ierr);
  ierr = VecGetArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(c->deta, &deta); CHKERRQ(ierr);
  for (int i=0; i<n; i++) {
    eta[i]  = 1.0 + ac->gamma * pow(u[i], ac->exponent);
    deta[i] = ac->exponent * ac->gamma * pow(u[i],ac->exponent-1.0);
    for (int d=0; d < c->d; d++) {
      w_[d][i]   = eta[i] * u_[d][i];
    }
  }
  ierr = VecRestoreArray(c->w[0], &u); CHKERRQ(ierr);
  ierr = VecRestoreArrays(c->gradu, c->d, &u_); CHKERRQ(ierr);
  ierr = VecRestoreArrays(&c->w[1], c->d, &w_); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->deta, &deta); CHKERRQ(ierr);

  ierr = VecZeroEntries(c->w[0]); CHKERRQ(ierr);
  for (int d=0; d < c->d; d++) {
    ierr = MatMult(c->D[d], c->w[1+d], c->w[1+c->d]); CHKERRQ(ierr);
    ierr = VecAXPY(c->w[0], -1.0, c->w[1+c->d]); CHKERRQ(ierr);
  }
#if DEBUG
  ierr = VecPrint2(c->w[0], c->dim[0], c->dim[1], "out function"); CHKERRQ(ierr); printf("\n");
#endif
  ierr = VecScatterBegin(c->scatterLG, c->w[0], rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[0], rhs, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecAXPY(rhs, -1.0, ac->b); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes, Vec w, Mat *A, Mat *P, MatStructure *flag, void *void_ac) {
  PetscErrorCode ierr;
  AppCtx *ac;
  MatElliptic *c;
  PetscScalar *eta, *deta, **u0_, *x;
  PetscInt *ixL;

  PetscFunctionBegin;
  // The nonlinear term has already been fixed up by FormFunction() so we just
  // need to deal with the preconditioner here.  In principle, this is the
  // Jacobian about `w', but we will rely on the viscosity already being
  // properly set since FormFunction() has already been called with the same `w'.
  ac = (AppCtx *)void_ac;
  ierr = MatShellGetContext(*A, (void **)&c); CHKERRQ(ierr);
  ierr = VecGetArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(c->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArrays(c->gradu, c->d, &u0_); CHKERRQ(ierr);
  ierr = VecGetArray(c->x, &x); CHKERRQ(ierr);
  ierr = ISGetIndices(c->isL, &ixL); CHKERRQ(ierr);
  {
    PetscInt J[2 * c->d + 1];
    PetscScalar v[2* c->d + 1];
    PetscInt k;
    PetscScalar x0, xMM, xPP, xM, idxM, xP, idxP, idx, eM, deM, du0M, eP, deP, du0P;
    for (BlockIt it = BlockIt(c->d, c->dim); !it.done; it.next()) { // loop over local dof
      const PetscInt i = it.i;
      if (ixL[i] < 0) continue; // Not a global dof
      J[0] = ixL[i]; v[0] = 0.0; k = 1;
      for (int j=0; j < c->d; j++) {
        const PetscInt iM = it.shift(j, -1);
        const PetscInt iP = it.shift(j,  1);
        if (iM < 0 || iP < 0) SETERRQ(1, "Local neighbor not on local grid.");
        x0 = x[i*c->d+j]; xMM = x[iM*c->d+j]; xPP = x[iP*c->d+j];
        xM = 0.5 * (xMM + x0); idxM = 1.0 / (x0 - xMM); xP = 0.5 * (x0 + xPP); idxP = 1.0 / (xPP - x0); idx = 1.0 / (xP - xM);
        eM = 0.5 * (eta[iM] + eta[i]); deM = 0.5 * (deta[iM] + deta[i]); du0M = 0.5 * (u0_[j][iM] + u0_[j][i]);
        eP = 0.5 * (eta[iP] + eta[i]); deP = 0.5 * (deta[iP] + deta[i]); du0P = 0.5 * (u0_[j][iP] + u0_[j][i]);
        J[k] = ixL[iM]; v[k] = -idx * (idxM * eM - 0.5 * deM * du0M); k++;
        J[k] = ixL[iP]; v[k] = -idx * (idxP * eP + 0.5 * deP * du0P); k++;
        v[0] += idx * (idxP * eP + idxM * eM - 0.5 * (deP * du0P - deM * du0M));
      }
      ierr = MatSetValues(*P, 1, J, k, J, v, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->deta, &deta); CHKERRQ(ierr);
  ierr = VecRestoreArrays(c->gradu, c->d, &u0_); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->x, &x); CHKERRQ(ierr);
  ierr = ISRestoreIndices(c->isL, &ixL); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateExactSolution"
PetscErrorCode CreateExactSolution(SNES snes, Vec u, Vec u2) {
  PetscErrorCode ierr;
  AppCtx *ac;
  MatElliptic *c;
  PetscScalar *X, *w0, *w1, *v, *w, z, s;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes, (void **)&ac); CHKERRQ(ierr);
  ierr = MatShellGetContext(ac->A, (void **)&c); CHKERRQ(ierr);
  const PetscInt d = c->d;
  const double gamma = ac->gamma, exponent = ac->exponent;
  s = 0.5;
  if (ac->exact == 0 || ac->exact == 3) {
    PetscReal cos_scale;
    ierr = PetscOptionsGetReal(PETSC_NULL, "-cos_scale", &cos_scale, PETSC_NULL);CHKERRQ(ierr);
    s *= cos_scale;
  }

  ierr = VecGetArray(c->w[0], &w0); CHKERRQ(ierr);
  ierr = VecGetArray(c->w[1], &w1); CHKERRQ(ierr);
  ierr = VecGetArray(c->x, &X); CHKERRQ(ierr);
  for (BlockIt it = BlockIt(c->d, c->dim); !it.done; it.next()) {
    const PetscInt i = it.i;
    const PetscScalar *x = &X[it.i*d];
    v = &w0[i], w = &w1[i];
    switch (ac->exact) {
      case 0: { // separable cosine, handles nonlinearity, zero on boundary iff cos_scale in 1,2,...
        v[0] = 1.0; w[0] = 0.0;
        for (int j=0; j < d; j++) v[0] *= cos(s*PI*x[j]);
        const double eta  = 1.0 + gamma * pow(v[0], exponent);
        const double deta = (PetscAbs(exponent) < 1e-10) ? 0.0 : gamma * exponent * pow(v[0], exponent-1.0);
        for (int j=0; j < d; j++) {
          double dv = 1.0;
          for (int k=0; k < d; k++) dv *= (k == j) ? -s*PI*sin(s*PI*x[k]) : cos(s*PI*x[k]);
          const double d2v = -PetscSqr(s*PI)*v[0];
          w[0] += deta * PetscSqr(dv) + eta * d2v;
        }
        w[0] = -w[0];
      } break;
      case 1: // separable quadratics, zero on boundary
        v[0] = 1.0; w[0] = 0.0;
        for (int j=0; j < d; j++) {
          v[0] *= (1 - x[j]) * (1 + x[j]);
          z = 1.0;
          for (int k=0; k < d; k++) {
            if (k != j) z *= 2.0 * (1 - x[k]) * (1 + x[k]);
          }
          w[0] += z;
        }
        break;
      case 2: // separable polynomials, nonzero on boundary
        v[0] = 1.0; w[0] = 0.0;
        for (int j=0; j < d; j++) {
          v[0] *= pow(x[j], 4+j);
          z = 1.0;
          for (int k=0; k < d; k++) {
            if (k == j) z *= (4+k) * (3+k) * pow(x[k], 2+k);
            else z *= pow(x[k], 4+k);
          }
          w[0] -= z;
        }
        break;
      default:
        SETERRQ(1, "Choose an exact solution.");
    }
  }

  ierr = VecRestoreArray(c->w[0], &w0); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->w[1], &w1); CHKERRQ(ierr);

  if (ac->debug) {
    ierr = VecPrint2(c->w[0], c->dim[0], c->dim[1], "exact w0"); CHKERRQ(ierr); printf("\n");
    ierr = VecPrint2(c->w[1], c->dim[0], c->dim[1], "exact w1"); CHKERRQ(ierr); printf("\n");
  }
  ierr = VecScatterBegin(c->scatterLG, c->w[0], u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[0], u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->w[1], u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[1], u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLD, c->w[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLD, c->w[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecCopy(u2, ac->b);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPrint2"
PetscErrorCode VecPrint2(Vec v, PetscInt m, PetscInt n, const char *name) {
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscScalar *x;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v, &comm); CHKERRQ(ierr);
  ierr = VecGetArray(v, &x); CHKERRQ(ierr);
  for (int j=0; j<n; j++) {
    ierr = PetscPrintf(comm, "%14s: ", name);CHKERRQ(ierr);
    for (int i=m-1; i>=0; i--) {
      ierr = PetscPrintf(comm, "%12.3e", x[i*n + j]); CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "\n"); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
