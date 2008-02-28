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
  KSP            ksp;      /* linear solver context */
  PC             pc;        /* preconditioner context */
  PetscReal      norm;      /* norm of solution error */
  PetscScalar    v,one = 1.0,none = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its, mn;
  PetscErrorCode ierr;
  PetscTruth     user_defined_pc, user_defined_mat;

  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);

  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = MatCreateChebD1(PETSC_COMM_WORLD, x, b,
                         FFTW_ESTIMATE | FFTW_PRESERVE_INPUT, &A);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  double *a;
  ierr = VecGetArray(u, &a); CHKERRQ(ierr);
  for (int i=0; i<m; i++) a[i] = exp(cos(i * PI / (m-1)));
  ierr = VecRestoreArray(u, &a); CHKERRQ(ierr);

  ierr = MatMult(A, u, b); CHKERRQ(ierr);
  // ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecCopy(b, x); CHKERRQ(ierr);

  ierr = VecAXPY(x,none,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  // ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %D\n",norm,its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A\n",norm);CHKERRQ(ierr);

  /* /\* ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); *\/ */
  /* printf("\n"); */
  /* ierr = VecView(b, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); */
  /* printf("\n"); */
  /* ierr = VecView(u, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); */

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
