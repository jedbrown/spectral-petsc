static char help[] = "Test of Chebyshev differentiation code.\n";

#define BC    1
#define SOLVE 1

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "chebyshev.h"
#include <petscksp.h>
#include <stdbool.h>

typedef struct {
  int rank;
  int *dim;
  Mat *A;
  Vec *u;
  Vec *v;
} MatPoisson;

extern PetscErrorCode MatPoissonCreate(MPI_Comm comm, int rank, int *dim, unsigned flag, Vec vx, Vec vy, Mat *A);
extern PetscErrorCode MatPoissonMult(Mat, Vec, Vec);
extern PetscErrorCode MatPoissonDestroy(Mat);
extern PetscErrorCode AssemblePoissonPC2(int m, int n, Mat *A);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x, b, u, u2;
  Mat            A, L, PL;
  KSP            ksp;
  // PC             pc;
  PetscReal      norm;
  PetscInt       m1 = 5, m = 8,n = 7, p=1, d=0, dd, its;
  PetscErrorCode ierr;
  // PetscTruth     user_defined_pc, user_defined_mat;

  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);

  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-m1",&m1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-d",&d,PETSC_NULL);CHKERRQ(ierr);

  if (n == 1) dd = 1;
  else if (p == 1) dd = 2;
  else dd = 3;

  int dim[] = { m, n, p };
  // ierr = VecCreate(PETSC_COMM_WORLD, &u2);CHKERRQ(ierr);
  // ierr = VecSetSizes(u2, PETSC_DECIDE, m * n);CHKERRQ(ierr);
  // ierr = VecSetFromOptions(u2);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD, m * n * p, &u); CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &u2);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &x);CHKERRQ(ierr);
  ierr = MatCreateCheb(PETSC_COMM_WORLD, dd, d, dim, FFTW_ESTIMATE, x, b, &A); CHKERRQ(ierr);
  ierr = MatPoissonCreate(PETSC_COMM_WORLD, dd, dim, FFTW_ESTIMATE, x, b, &L); CHKERRQ(ierr);
  ierr = AssemblePoissonPC2(m, n, &PL); CHKERRQ(ierr);

  if (false) {
    PetscTruth flag;
    ierr = MatIsSymmetric(PL, 1e-4, &flag); CHKERRQ(ierr);
    printf("Symmetric?  %d\n", flag);
  }


  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,L,PL,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  // 3-D solution function
  double *uu, *uu2;
  ierr = VecGetArray(u, &uu); CHKERRQ(ierr);
  ierr = VecGetArray(u2, &uu2); CHKERRQ(ierr);
  for (int i=0; i < m; i++) {
    double x = (m==1) ? 0 : cos (i * PI / (m-1));
    for (int j=0; j < n; j++) {
      double y = (n==1) ? 0 : cos (j * PI / (n-1));
      for (int k=0; k < p; k++) {
        // double z = (p==1) ? 0 : cos (k * PI / (p-1));
        int ix = (i*n + j) * p + k;
        uu[ix] = cos(0.5 * PI * x) * cos (0.5 * PI * y);
        uu2[ix] = 0.5 * PI * PI * uu[ix];
        // uu[ix] = pow(x, 4) * pow(y, 3) + pow(y, 5);
        // uu2[ix] = -(4 * 3 * pow(x, 2) * pow(y, 3) + 3 * 2 * pow(x,4) * y + 5 * 4 * pow(y, 3));
        //uu[ix] =
      }
    }
  }
  ierr = VecRestoreArray(u, &uu); CHKERRQ(ierr);
  ierr = VecRestoreArray(u2, &uu2); CHKERRQ(ierr);

  ierr = VecCopy(u2, b); CHKERRQ(ierr);

#if 0
  ierr = MatMult(PL, u, x); CHKERRQ(ierr);
  //ierr = VecCopy(b, x); CHKERRQ(ierr);
  ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  printf("\n");
  ierr = VecView(u2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  printf("\n");
  ierr = VecAXPY(x, -1.0, u2);CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A\n",norm);CHKERRQ(ierr);
#endif

  // ierr = MatMult(L, u, b); CHKERRQ(ierr);

#if BC
  { // Set up boundary conditions
    double *uu, *bb;
    const int m = dim[0], n = dim[1];
    ierr = VecGetArray(u, &uu); CHKERRQ(ierr);
    ierr = VecGetArray(b, &bb); CHKERRQ(ierr);
    for (int i = 0; i < m; i++) {
      int ix0 = i * n + 0;
      int ixn = i * n + n - 1;
      bb[ix0] = uu[ix0];
      bb[ixn] = uu[ixn];
    }
    for (int j = 0; j < n; j++) {
      int ix0 = 1 * n + j;
      int ixn = (m-2) * n + j;
      bb[ix0] = uu[ix0];
      bb[ixn] = uu[ixn];
    }
    ierr = VecRestoreArray(u, &uu); CHKERRQ(ierr);
    ierr = VecRestoreArray(b, &bb); CHKERRQ(ierr);
  }
#endif

#if SOLVE
  ierr = VecSet(x, 0.0); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

  /* ierr = VecView(b, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); */
  /* printf("\n"); */
  /* ierr = VecView(u, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); */
  /* printf("\n"); */
  /* ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); */
  /* printf("\n"); */

  ierr = VecAXPY(x, -1.0, u); CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %A iterations %D\n", norm, its); CHKERRQ(ierr);
  // ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A\n",norm);CHKERRQ(ierr);
#endif

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);

  /* if (user_defined_pc) { */
  /*   ierr = SampleShellPCDestroy(shell);CHKERRQ(ierr); */
  /* } */

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;

}

#undef __FUNCT__
#define __FUNCT__ "MatPoissonCreate"
PetscErrorCode MatPoissonCreate(MPI_Comm comm, int rank, int *dim, unsigned flag, Vec vx, Vec vy, Mat *A) {
  PetscErrorCode ierr;
  MatPoisson *c;
  int n;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(MatPoisson), &c); CHKERRQ(ierr);
  ierr = PetscMalloc(rank * sizeof(int), &(c->dim)); CHKERRQ(ierr);
  ierr = PetscMalloc(rank * sizeof(Mat *), &(c->A)); CHKERRQ(ierr);
  c->rank = rank;

  for (int i=0; i < rank; i++) {
    c->dim[i] = dim[i];
    ierr = MatCreateCheb(comm, rank, i, dim, flag, vx, vy, &(c->A[i])); CHKERRQ(ierr);
  }

  ierr = VecDuplicateVecs(vx, rank, &c->u); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(vx, rank, &c->v); CHKERRQ(ierr);

  ierr = VecGetSize(vx, &n); CHKERRQ(ierr);

  ierr = MatCreateShell(comm, n, n, n, n, c, A); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))MatPoissonMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))MatPoissonDestroy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPoissonMult"
PetscErrorCode MatPoissonMult(Mat A, Vec x, Vec y) {
  PetscErrorCode ierr;
  MatPoisson *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void *)&c); CHKERRQ(ierr);
  for (int i = 0; i < c->rank; i++) {
    ierr = MatMult(c->A[i], x, c->u[i]); CHKERRQ(ierr);
    ierr = MatMult(c->A[i], c->u[i], c->v[i]); CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  for (int i = 0; i < c->rank; i++) {
    ierr = VecAXPY(y, -1.0, c->v[i]); CHKERRQ(ierr);
  }

#if BC
  { // Fix one point as Dirichlet, everything else is homogenous Neumann
    PetscScalar *xx, *yy;
    const int m = c->dim[0], n = c->dim[1];
    ierr = VecGetArray(x, &xx); CHKERRQ(ierr);
    ierr = VecGetArray(y, &yy); CHKERRQ(ierr);
    for (int i = 0; i < m; i++) {
      const int ix0 = i * n + 0;
      int ixn = i * n + n - 1;
      yy[ix0] = xx[ix0];
      yy[ixn] = xx[ixn];
    }
    for (int j = 0; j < n; j++) {
      int ix0 = j;
      int ixn = (m-1) * n + j;
      yy[ix0] = xx[ix0];
      yy[ixn] = xx[ixn];
    }
    ierr = VecRestoreArray(x, &xx); CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &yy); CHKERRQ(ierr);
  }
#endif

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPoissonDestroy"
PetscErrorCode MatPoissonDestroy (Mat A) {
  PetscErrorCode ierr;
  MatPoisson *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void *)&c); CHKERRQ(ierr);
  for (int i = 0; i < c->rank; i++) {
    ierr = MatDestroy(c->A[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(c->A); CHKERRQ(ierr);
  ierr = PetscFree(c->dim); CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->u, c->rank); CHKERRQ(ierr);
  ierr = VecDestroyVecs(c->v, c->rank); CHKERRQ(ierr);
  ierr = PetscFree(c);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssemblePoissonPC2"
PetscErrorCode AssemblePoissonPC2(int m, int n, Mat *A) {
  PetscErrorCode ierr;
  double *x, *y;

  PetscFunctionBegin;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m*n, m*n, 5, PETSC_NULL, A); CHKERRQ(ierr);
  ierr = PetscMalloc2(m, double, &x, n, double, &y); CHKERRQ(ierr);
  for (int i = 0; i < m; i++) x[i] = cos(i * PI / (m - 1));
  for (int j = 0; j < n; j++) y[j] = cos(j * PI / (n - 1));

  // Strong Dirichlet everywhere, then overwrite with correct values in the interior.
  for (int I = 0; I < m*n; I++) {
    ierr = MatSetValue(*A, I, I, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);

  int J[5];
  double v[5];
  for (int i = 1; i < m - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      const int I = i * n + j;
      const bool f = true;
      // const double X = x[I], Y = y[I];
      // J[0] = I - n; J[1] = I - 1; J[2] = I; J[3] = I + 1; J[4] = I + n;
      int k=0;
      if (f && i != 1) { J[k] = I-n; v[k++] = -1.0 / ((x[i-1] - x[i]) * (x[i-1] - x[i+1])); }
      if (f && j != 1) { J[k] = I-1; v[k++] = -1.0 / ((y[j-1] - y[j]) * (y[j-1] - y[j+1])); }
      J[k] = I; v[k++] = -(  1.0 / ((x[i] - x[i-1]) * (x[i] - x[i+1]))
                           + 1.0 / ((y[j] - y[j-1]) * (y[j] - y[j+1])));
      if (f && j != n-2) { J[k] = I+1; v[k++] = -1.0 / ((y[j+1] - y[j-1]) * (y[j+1] - y[j])); }
      if (f && i != m-2) { J[k] = I+n; v[k++] = -1.0 / ((x[i+1] - x[i-1]) * (x[i+1] - x[i])); }
      ierr = MatSetValues(*A, 1, &I, k, J, v, INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree2(x, y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
