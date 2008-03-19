static char help[] = "A nonlinear elliptic equation by Chebyshev differentiation.\n";

#define SOLVE 1
#define CHECK_EXACT 1

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
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
PetscErrorCode LiftDirichlet_Elliptic(Mat A, Vec x, Vec b);
PetscErrorCode DirichletBdy(int d, double *x, double *n, BdyCond *bc);
PetscErrorCode CreateExactSolution(SNES snes, Vec u, Vec u2, Vec b, Vec b0);
PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
PetscErrorCode FormJacobian(SNES, Vec, Mat *, Mat *, MatStructure *, void *);
PetscErrorCode VecPrint2(Vec, PetscInt, PetscInt, const char *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  MPI_Comm       comm;
  Vec            x, r, b, b0, u, u2;
  Mat            A, P;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  PetscReal      norm;
  PetscInt       m, n, p, d, its, exact;
  SNESConvergedReason reason;
  PetscErrorCode ierr;
  // MatElliptic *c;
  AppCtx *ac;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);
  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(AppCtx), &ac); CHKERRQ(ierr);
  ierr = PetscMalloc(3*sizeof(PetscInt), &ac->dim); CHKERRQ(ierr);

  m = 4; n = 5; p = 1; exact = 0; ac->gamma = 0.0; ac->exponent = 2.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elliptic problem options", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "x dim extent", "elliptic.c", m, &m, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "y dim extent", "elliptic.c", n, &n, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "z dim extent", "elliptic.c", p, &p, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-exact", "exact solution type", "elliptic.c", exact, &exact, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-gamma", "strength of nonlinearity", "elliptic.c", ac->gamma, &ac->gamma, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-exponent", "exponent of nonlinearity", "elliptic.c", ac->exponent, &ac->exponent, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  if (n == 1) d = 1;
  else if (p == 1) d = 2;
  else d = 3;
  ac->d = d;
  ac->dim[0] = m; ac->dim[1] = n; ac->dim[2] = p;
  ac->exact = exact;

  //ierr = VecCreate
  ierr = MatCreate_Elliptic(PETSC_COMM_WORLD, d, ac->dim, FFTW_ESTIMATE, DirichletBdy, &u, &A); CHKERRQ(ierr);
  {
    PetscInt m,n;
    ierr = MatGetSize(A, &m, &n); CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF, &P); CHKERRQ(ierr);
    ierr = MatSetSizes(P, m, n, m, n); CHKERRQ(ierr);
    ierr = MatSetType(P, MATUMFPACK); CHKERRQ(ierr);
    ierr = MatSetFromOptions(P); CHKERRQ(ierr);
    //ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 5, PETSC_NULL, &P); CHKERRQ(ierr);
  }

  //ierr = VecPrint2(ac->x, m, n*2, "coordinates"); CHKERRQ(ierr);

  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b0);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &u2);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &ac->b);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, P, FormJacobian, ac); CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, FormFunction, ac); CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes, ac); CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
  ac->A = A;
  // u = exact solution, u2 = A(u) u
  // b = RHS with dirichlet, b0 = just dirichlet
  ierr = CreateExactSolution(snes,u,u2,b,b0);CHKERRQ(ierr);
  ierr = VecCopy(u2,ac->b);CHKERRQ(ierr);

#if CHECK_EXACT
  //ierr = VecSet(r, 0.0); CHKERRQ(ierr);
  //ierr = FormFunction(snes, r, x, ac); CHKERRQ(ierr);
  ierr = FormFunction(snes, u, r, ac); CHKERRQ(ierr);
#if DEBUG
  ierr = VecPrint2(u, m-2, n-2, "exact u"); CHKERRQ(ierr); printf("\n");
  ierr = VecPrint2(u2, m-2, n-2, "exact u2"); CHKERRQ(ierr); printf("\n");
  ierr = VecPrint2(b, m-2, n-2, "discrete b"); CHKERRQ(ierr); printf("\n");
  ierr = VecPrint2(b0, m-2, n-2, "discrete b0"); CHKERRQ(ierr); printf("\n");
  ierr = VecPrint2(r, m-2, n-2, "discrete residual"); CHKERRQ(ierr); printf("\n");
#endif
  ierr = VecNorm(r, NORM_INFINITY, &norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of residual %A\n",norm);CHKERRQ(ierr);
#endif

#if SOLVE
  ierr = VecSet(x, 0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes, PETSC_NULL, x); CHKERRQ(ierr);

  //ierr = VecPrint2(u, m-2, n-2, "exact u"); CHKERRQ(ierr); printf("\n");
  //ierr = VecPrint2(x, m-2, n-2, "computed u"); CHKERRQ(ierr);

  ierr = VecAXPY(x, -1.0, u); CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of nonlinear iterations = %D\n", its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Norm of error = %A\n", norm);CHKERRQ(ierr);
  /* ierr = SNESGetKSP(snes, ksp); CHKERRQ(ierr); */
  /* ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr); */
  /* ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %A iterations %D\n", norm, its); CHKERRQ(ierr); */
  //ierr = VecPrint2(b0, m-2, n-2, "b0"); CHKERRQ(ierr);
#endif

  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(u2);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = VecDestroy(b0);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);

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
  ierr = PetscMalloc2(d, PetscInt, &c->dim, d, Vec, &c->D); CHKERRQ(ierr);
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
  ierr = PetscMalloc6(m, PetscInt, &ixL, m, PetscInt, &ixG, 2*sumInt(c->d,c->dim), PetscInt, &ixD,
                      c->d, PetscInt, &ind, c->d, PetscScalar, &x, c->d, PetscScalar, &n); CHKERRQ(ierr);
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

  ierr = VecCreateSeq(comm, g, &vG); CHKERRQ(ierr);
  ierr = VecCreateSeq(comm, d, &vD); CHKERRQ(ierr);
  { // Fill dirichlet values into the special dirichlet vector.
    PetscScalar *v;
    ierr = VecGetArray(vD, &v); CHKERRQ(ierr);
    for (int i=0; i<d; i++) v[i] = uD[i];
    ierr = VecRestoreArray(vD, &v); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->w[1], &uD); CHKERRQ(ierr);

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

  ierr = PetscFree6(ixL, ixG, ixD, ind, x, n); CHKERRQ(ierr);
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
#define __FUNCT__ "LiftDirichlet_Elliptic"
PetscErrorCode LiftDirichlet_Elliptic(Mat A, Vec x, Vec b) {
  PetscErrorCode ierr;
  MatElliptic *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = VecCopy(c->dirichlet, c->dirichlet0); CHKERRQ(ierr);
  ierr = MatMult(A, x, b); CHKERRQ(ierr);
  ierr = VecZeroEntries(c->dirichlet0); CHKERRQ(ierr);
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
  PetscScalar *eta, *deta, *u0x, *u0y, *x;
  PetscInt n, *ixL, *ixG;

  PetscFunctionBegin;
  // The nonlinear term has already been fixed up by FormFunction() so we just need to deal with the preconditioner here.
  ac = (AppCtx *)void_ac;
  ierr = MatShellGetContext(*A, (void **)&c); CHKERRQ(ierr);
  ierr = VecGetArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(c->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArray(c->gradu[0], &u0x); CHKERRQ(ierr);
  ierr = VecGetArray(c->gradu[1], &u0y); CHKERRQ(ierr);
  ierr = VecGetArray(c->x, &x); CHKERRQ(ierr);
  ierr = ISGetLocalSize(c->isG, &n); CHKERRQ(ierr);
  ierr = ISGetIndices(c->isG, &ixG); CHKERRQ(ierr);
  ierr = ISGetIndices(c->isL, &ixL); CHKERRQ(ierr);

  PetscInt J[5], k, l, d;
  PetscScalar v[5], x0, y0, e0, de0, u0x0, u0y0, xE, dxE, eE, deE, u0xE, xW, dxW, eW, deW, u0xW, yN, dyN, eN, deN, u0yN, yS, dyS, eS, deS, u0yS, idxE, idxW, idx, idyN, idyS, idy;
  d = c->d;
  for (int I=0; I<n; I++) { // loop over global degrees of freedom
    const int iL = ixG[I];  // The local index
    const int N = productInt(c->d, c->dim) / c->dim[0];
    //const int i = iL / (c->dim[1] * c->dim[2]); // Only two dimensions work now.
    //const int j = (iL / c->dim[2]) % c->dim[1];
    // This is broken if we are not using a dirichlet boundary.
    l = iL;   x0 = x[l*d  ]; y0 = x[l*d+1]; e0 = eta[l]; de0 = deta[l]; u0x0 = u0x[l]; u0y0 = u0y[l];
    l = iL-N; xE = x[l*d  ] + x0; dxE = x[l*d  ] - x0; eE = eta[l] + e0; deE = deta[l] + de0; u0xE = 0.5*(u0x[l] + u0x0);
    l = iL-1; yN = x[l*d+1] + y0; dyN = x[l*d+1] - y0; eN = eta[l] + e0; deN = deta[l] + de0; u0yN = 0.5*(u0y[l] + u0y0);
    l = iL+1; yS = x[l*d+1] + y0; dyS = y0 - x[l*d+1]; eS = eta[l] + e0; deS = deta[l] + de0; u0yS = 0.5*(u0y[l] + u0y0);
    l = iL+N; xW = x[l*d  ] + x0; dxW = x0 - x[l*d  ]; eW = eta[l] + e0; deW = deta[l] + de0; u0xW = 0.5*(u0x[l] + u0x0);
    // printf("%8f %8f %8f   %8f %8f %8f\n", xW, x0, xE, yS, y0, yN);
    idx = 1.0/(xE-xW); idxE = 1.0/dxE; idxW = 1.0/dxW;
    idy = 1.0/(yN-yS); idyN = 1.0/dyN; idyS = 1.0/dyS;
    k = 0;
    if ((J[k] = ixL[iL-N]) >= 0) { v[k] = -idx*(idxE*eE + 0.5*deE*u0xE); k++; }
    if ((J[k] = ixL[iL-1]) >= 0) { v[k] = -idy*(idyN*eN + 0.5*deN*u0yN); k++; }
    J[k] = ixL[iL]; v[k] = idx*(idxE*eE - 0.5*deE*u0xE + idxW*eW + 0.5*deW*u0xW) + idy*(idyN*eN - 0.5*deN*u0yN + idyS*eS + 0.5*deS*u0yS); k++;
    if ((J[k] = ixL[iL+1]) >= 0) { v[k] = -idy*(idyS*eS - 0.5*deS*u0yS); k++; }
    if ((J[k] = ixL[iL+N]) >= 0) { v[k] = -idx*(idxW*eW - 0.5*deW*u0xW); k++; }
    ierr = MatSetValues(*P, 1, &I, k, J, v, INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->eta, &eta); CHKERRQ(ierr);
  ierr = VecGetArray(c->deta, &deta); CHKERRQ(ierr);
  ierr = VecGetArray(c->gradu[0], &u0x); CHKERRQ(ierr);
  ierr = VecGetArray(c->gradu[1], &u0y); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->x, &x); CHKERRQ(ierr);
  ierr = ISRestoreIndices(c->isG, &ixG); CHKERRQ(ierr);
  ierr = ISRestoreIndices(c->isL, &ixL); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateExactSolution"
PetscErrorCode CreateExactSolution(SNES snes, Vec u, Vec u2, Vec b, Vec b0) {
  PetscErrorCode ierr;
  AppCtx *ac;
  MatElliptic *c;
  PetscInt m, n, p;

  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes, (void **)&ac); CHKERRQ(ierr);
  ierr = MatShellGetContext(ac->A, (void **)&c); CHKERRQ(ierr);
  m = c->dim[0];
  n = (c->d == 1) ? 1 : c->dim[1];
  p = (c->d <= 2) ? 1 : c->dim[2];
  double *w0, *w1;
  ierr = VecGetArray(c->w[0], &w0); CHKERRQ(ierr);
  ierr = VecGetArray(c->w[1], &w1); CHKERRQ(ierr);
  for (int i=0; i < m; i++) {
    double x = (m==1) ? 0 : cos (i * PI / (m-1));
    for (int j=0; j < n; j++) {
      double y = (n==1) ? 0 : cos (j * PI / (n-1));
      for (int k=0; k < p; k++) {
        // double z = (p==1) ? 0 : cos (k * PI / (p-1));
        int ix = (i*n + j) * p + k;
        switch (ac->exact) {
          case 0:
            w0[ix] = cos(0.5 * PI * x) * cos (0.5 * PI * y);
            w1[ix] = 0.5 * PI * PI * w0[ix];
            break;
          case 1:
            w0[ix] = (x-1) * (x+1) * (y-1) * (y+1);
            w1[ix] = -2.0 * ((x-1) * (x+1) + (y-1) * (y+1));
            break;
          case 2:
            w0[ix] = pow(x, 5) * pow(y,6);
            w1[ix] = -(5*4*pow(x,3)*pow(y,6) + 6*5*pow(x,5)*pow(y,4));
            break;
          case 3:
            w0[ix] = exp(2*x) * exp(3*y);
            w1[ix] = -13 * w0[ix];
            break;
          case 4:
            w0[ix] = sin(10*x) + sin(12*y);
            w1[ix] = 100 * sin(10*x) + 144*sin(12*y);
            break;
          case 5:
            w0[ix] = exp(-(x*x + y*y));
            w1[ix] = (4 - 4*x*x - 4*y*y) * w0[ix];
            break;
          default:
            SETERRQ(1, "Choose an exact solution.");
        }
        // uu[ix] = pow(x, 4) * pow(y, 3) + pow(y, 5);
        // uu2[ix] = -(4 * 3 * pow(x, 2) * pow(y, 3) + 3 * 2 * pow(x,4) * y + 5 * 4 * pow(y, 3));
        //uu[ix] =
      }
    }
  }
  ierr = VecRestoreArray(c->w[0], &w0); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->w[1], &w1); CHKERRQ(ierr);
#if DEBUG
  ierr = VecPrint2(c->w[0], m, n, "exact w0"); CHKERRQ(ierr); printf("\n");
  ierr = VecPrint2(c->w[1], m, n, "exact w1"); CHKERRQ(ierr); printf("\n");
#endif
  ierr = VecScatterBegin(c->scatterLG, c->w[0], u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[0], u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->w[1], u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w[1], u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLD, c->w[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLD, c->w[0], c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  ierr = VecSet(b, 0.0); CHKERRQ(ierr); // b is the inhomogenous dirichlet part, zero at interior nodes
  ierr = FormFunction(snes, b, b0, (void *)ac); CHKERRQ(ierr);
  //ierr = LiftDirichlet_Elliptic(ac, FormFunction, b, b0); CHKERRQ(ierr); // Put it's contribution into b0
  ierr = VecCopy(u2, b); CHKERRQ(ierr); // Put exact right hand side in b
  ierr = VecAXPY(b, -1.0, b0); CHKERRQ(ierr); // Lift inhomogenous part to right hand side

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
