static char help[] = "Test of Chebyshev differentiation code.\n";

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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;   /* approx solution, RHS, exact solution */
  Mat            A;
  Vec            x2,b2,u2;   /* approx solution, RHS, exact solution */
  Mat            A2;
  KSP            ksp;      /* linear solver context */
  PC             pc;        /* preconditioner context */
  PetscReal      norm;      /* norm of solution error */
  PetscScalar    v,one = 1.0,none = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend, m1 = 5, m = 8,n = 7, p=1, d=0, its;
  PetscErrorCode ierr;
  PetscTruth     user_defined_pc, user_defined_mat;

  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);

  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-m1",&m1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-d",&d,PETSC_NULL);CHKERRQ(ierr);

  // ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  // ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
  // ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD, m1, &u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = MatCreateChebD1(PETSC_COMM_WORLD, x, b,
                         FFTW_ESTIMATE | FFTW_PRESERVE_INPUT, &A); CHKERRQ(ierr);

  int dims[] = { m, n, p };
  // ierr = VecCreate(PETSC_COMM_WORLD, &u2);CHKERRQ(ierr);
  // ierr = VecSetSizes(u2, PETSC_DECIDE, m * n);CHKERRQ(ierr);
  // ierr = VecSetFromOptions(u2);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD, m * n * p, &u2); CHKERRQ(ierr);
  ierr = VecDuplicate(u2, &b2);CHKERRQ(ierr);
  ierr = VecDuplicate(b2, &x2);CHKERRQ(ierr);
  ierr = MatCreateCheb(PETSC_COMM_WORLD, 3, d, dims, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT,
                       x2, b2, &A2); CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  // Solution function
  double *a;
  ierr = VecGetArray(u, &a); CHKERRQ(ierr);
  for (int i=0; i<m1; i++) {
    a[i] = exp(cos(i * PI / (m1-1)));
  }
  ierr = VecRestoreArray(u, &a); CHKERRQ(ierr);

  // 2-D solution function
  ierr = VecGetArray(u2, &a); CHKERRQ(ierr);
  for (int i=0; i < m; i++) {
    double x = (m==1) ? 0 : cos (i * PI / (m-1));
    for (int j=0; j < n; j++) {
      double y = (n==1) ? 0 : cos (j * PI / (n-1));
      for (int k=0; k < p; k++) {
        double z = (p==1) ? 0 : cos (k * PI / (p-1));
        a[(i*n + j) * p + k] = exp(x) + exp(y) + exp(z);
      }
    }
  }
  ierr = VecRestoreArray(u2, &a); CHKERRQ(ierr);

  ierr = MatMult(A, u, b); CHKERRQ(ierr);
  ierr = MatMult(A2, u2, b2); CHKERRQ(ierr);
  // ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecCopy(b, x); CHKERRQ(ierr);

  ierr = VecAXPY(x,none,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  // ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %D\n",norm,its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A\n",norm);CHKERRQ(ierr);

  /* /\* ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); *\/ */
  printf("\n");
  ierr = VecView(b, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  printf("\n");
  ierr = VecView(b2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

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
