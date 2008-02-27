
static char help[] = "Solves a linear system in parallel with KSP.  Also\n\
illustrates setting a user-defined shell preconditioner and using the\n\
macro __FUNCT__ to define routine names for use in error handling.\n\
Input parameters include:\n\
  -user_defined_pc : Activate a user-defined preconditioner\n\
  -user_defined_pc : Activate a user-defined matrix\n\n";

/*T
   Concepts: KSP^basic parallel example
   Concepts: PC^setting a user-defined shell preconditioner
   Concepts: error handling^Using the macro __FUNCT__ to define routine names;
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscksp.h"

/* Define context for user-provided preconditioner */
typedef struct {
  Vec diag;
} SampleShellPC;

typedef struct {
  PetscInt m, n;
} MatShellCtx;

/* Declare routines for user-provided preconditioner */
extern PetscErrorCode SampleShellPCCreate(SampleShellPC**);
extern PetscErrorCode SampleShellPCSetUp(SampleShellPC*,Mat,Vec);
extern PetscErrorCode SampleShellPCApply(void*,Vec x,Vec y);
extern PetscErrorCode SampleShellPCDestroy(SampleShellPC*);

extern PetscErrorCode MatShellMult(Mat, Vec, Vec);
extern PetscErrorCode MatShellMult2(Mat, Vec, Vec);
extern PetscErrorCode MatShellMult3(Mat, Vec, Vec);
extern PetscErrorCode MatShellGetDiagonal(Mat, Vec);
extern PetscErrorCode MatShellGetDiagonal2(Mat, Vec);
extern PetscErrorCode MatShellGetDiagonal3(Mat, Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;   /* approx solution, RHS, exact solution */
  Mat            A, B, B2, B3;         /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PC             pc;        /* preconditioner context */
  PetscReal      norm;      /* norm of solution error */
  SampleShellPC  *shell;    /* user-defined preconditioner context */
  MatShellCtx    mat_ctx;
  PetscScalar    v,one = 1.0,none = -1.0;
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its, mn;
  PetscErrorCode ierr;
  PetscTruth     user_defined_pc, user_defined_mat;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);


  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  mat_ctx.m = m; mat_ctx.n = n; mn = m * n;
  ierr = MatCreateShell(PETSC_COMM_WORLD, mn, mn, mn, mn, &mat_ctx, &B); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B, MATOP_MULT, (void(*)(void))MatShellMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B, MATOP_GET_DIAGONAL, (void(*)(void))MatShellGetDiagonal); CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD, mn, mn, mn, mn, &mat_ctx, &B2); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B2, MATOP_MULT, (void(*)(void))MatShellMult2); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B2, MATOP_GET_DIAGONAL, (void(*)(void))MatShellGetDiagonal2); CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD, mn, mn, mn, mn, &mat_ctx, &B3); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B3, MATOP_MULT, (void(*)(void))MatShellMult3); CHKERRQ(ierr);
  ierr = MatShellSetOperation(B3, MATOP_GET_DIAGONAL, (void(*)(void))MatShellGetDiagonal3); CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-user_defined_mat",&user_defined_mat); CHKERRQ(ierr);
  /* if (user_defined_mat) { A = B; } else { B = A; } */

  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = MatMult(B3,u,b);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,B3,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
         PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-user_defined_pc",&user_defined_pc);CHKERRQ(ierr);
  if (user_defined_pc) {
    Mat M, P;
    ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
    ierr = SampleShellPCCreate(&shell);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc,SampleShellPCApply);CHKERRQ(ierr);
    ierr = PCShellSetContext(pc,shell);CHKERRQ(ierr);
    ierr = PCShellSetName(pc,"MyPreconditioner");CHKERRQ(ierr);
    ierr = PCGetOperators(pc, &M, &P, PETSC_NULL); CHKERRQ(ierr);
    ierr = SampleShellPCSetUp(shell,P,x);CHKERRQ(ierr);
  } else {
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    // ierr = PCSetType(pc,PCSPAI);CHKERRQ(ierr);
  }


  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = VecAXPY(x,none,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %D\n",norm,its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);

  if (user_defined_pc) {
    ierr = SampleShellPCDestroy(shell);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;

}


#undef __FUNCT__
#define __FUNCT__ "SampleShellPCCreate"
/*
   SampleShellPCCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode SampleShellPCCreate(SampleShellPC **shell)
{
  SampleShellPC  *newctx;
  PetscErrorCode ierr;

  ierr         = PetscNew(SampleShellPC,&newctx);CHKERRQ(ierr);
  newctx->diag = 0;
  *shell       = newctx;
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SampleShellPCSetUp"
/*
   SampleShellPCSetUp - This routine sets up a user-defined
   preconditioner context.

   Input Parameters:
.  shell - user-defined preconditioner context
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  shell - fully set up user-defined preconditioner context

   Notes:
   In this example, we define the shell preconditioner to be Jacobi's
   method.  Thus, here we create a work vector for storing the reciprocal
   of the diagonal of the preconditioner matrix; this vector is then
   used within the routine SampleShellPCApply().
*/
PetscErrorCode SampleShellPCSetUp(SampleShellPC *shell,Mat pmat,Vec x)
{
  Vec            diag;
  PetscErrorCode ierr;

  ierr = VecDuplicate(x,&diag);CHKERRQ(ierr);
  ierr = MatGetDiagonal(pmat,diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  shell->diag = diag;

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SampleShellPCApply"
/*
   SampleShellPCApply - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
.  ctx - optional user-defined context, as set by PCShellSetContext()
.  x - input vector

   Output Parameter:
.  y - preconditioned vector

   Notes:
   Note that the PCSHELL preconditioner passes a void pointer as the
   first input argument.  This can be cast to be the whatever the user
   has set (via PCSetShellApply()) the application-defined context to be.

   This code implements the Jacobi preconditioner, merely as an
   example of working with a PCSHELL.  Note that the Jacobi method
   is already provided within PETSc.
*/
PetscErrorCode SampleShellPCApply(void *ctx,Vec x,Vec y)
{
  SampleShellPC   *shell = (SampleShellPC*)ctx;
  PetscErrorCode  ierr;

  ierr = VecPointwiseMult(y,x,shell->diag);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SampleShellPCDestroy"
/*
   SampleShellPCDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode SampleShellPCDestroy(SampleShellPC *shell)
{
  PetscErrorCode ierr;

  ierr = VecDestroy(shell->diag);CHKERRQ(ierr);
  ierr = PetscFree(shell);CHKERRQ(ierr);

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "MatShellMult"
PetscErrorCode MatShellMult(Mat A, Vec vx, Vec vy)
{
  PetscErrorCode ierr;
  PetscScalar *x, *y, v, four, one;
  PetscInt m, n, i, j, I;
  MatShellCtx *ctx;
  PetscTruth munge;

  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-munge",&munge); CHKERRQ(ierr);
  if (munge) {
    four = 5.0; one = 1.25;
  } else {
    four = 4.0; one = 1.0;
  }

  ierr = MatShellGetContext(A, (void **)&ctx); CHKERRQ(ierr);
  m = ctx->m;
  n = ctx->n;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      I = i * n + j;
      v = four * x[I];
      if (i > 0)     v -= one * x[I-n];
      if (i < m - 1) v -= one * x[I+n];
      if (j > 0)     v -= one * x[I-1];
      if (j < n - 1) v -= one * x[I+1];
      y[I] = v;
    }
  }

  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatShellGetDiagonal"
PetscErrorCode MatShellGetDiagonal(Mat A, Vec vx){
  PetscErrorCode ierr;
  PetscTruth munge;
  PetscScalar *x;

  ierr = PetscOptionsHasName(PETSC_NULL,"-munge",&munge); CHKERRQ(ierr);

  ierr = VecSet(vx, 4.0); CHKERRQ(ierr);

  if (munge) {
    ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
    x[0] = 2; x[1] = 5.0; x[2] = 10;
    ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  }

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatShellMult2"
PetscErrorCode MatShellMult2(Mat A, Vec vx, Vec vy)
{
  PetscErrorCode ierr;
  PetscScalar *x, *y, v, c0, c1, c2;
  PetscInt m, n, i, j, I;
  MatShellCtx *ctx;

  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  c0 = 2.5; c1 = - 4.0 / 3.0; c2 = 1.0 / 12.0;

  ierr = MatShellGetContext(A, (void **)&ctx); CHKERRQ(ierr);
  m = ctx->m;
  n = ctx->n;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      I = i * n + j;
      v = 2 * c0 * x[I];
      if (i > 1)     v += c2 * x[I-2*n];
      if (i > 0)     v += c1 * x[I-n];
      if (i < m - 2) v += c2 * x[I+2*n];
      if (i < m - 1) v += c1 * x[I+n];
      if (j > 1)     v += c2 * x[I-2];
      if (j > 0)     v += c1 * x[I-1];
      if (j < n - 2) v += c2 * x[I+2];
      if (j < n - 1) v += c1 * x[I+1];
      y[I] = v;
    }
  }

  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "MatShellGetDiagonal2"
PetscErrorCode MatShellGetDiagonal2(Mat A, Vec vx){
  PetscErrorCode ierr;

  ierr = VecSet(vx, 5.0); CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatShellMult3"
PetscErrorCode MatShellMult3(Mat A, Vec vx, Vec vy)
{
  PetscErrorCode ierr;
  PetscScalar *x, *y, v, c0, c1, c2, c3;
  PetscInt m, n, i, j, I;
  MatShellCtx *ctx;

  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  c0 = 49.0 / 18.0; c1 = - 1.5; c2 = 3.0 / 20.0; c3 = -1.0 / 90.0;

  ierr = MatShellGetContext(A, (void **)&ctx); CHKERRQ(ierr);
  m = ctx->m;
  n = ctx->n;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      I = i * n + j;
      v = 2 * c0 * x[I];
      if (i > 2)     v += c3 * x[I-3*n];
      if (i > 1)     v += c2 * x[I-2*n];
      if (i > 0)     v += c1 * x[I-n];
      if (i < m - 3) v += c3 * x[I+3*n];
      if (i < m - 2) v += c2 * x[I+2*n];
      if (i < m - 1) v += c1 * x[I+n];
      if (j > 2)     v += c3 * x[I-3];
      if (j > 1)     v += c2 * x[I-2];
      if (j > 0)     v += c1 * x[I-1];
      if (j < n - 3) v += c3 * x[I+3];
      if (j < n - 2) v += c2 * x[I+2];
      if (j < n - 1) v += c1 * x[I+1];
      y[I] = v;
    }
  }

  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "MatShellGetDiagonal3"
PetscErrorCode MatShellGetDiagonal3(Mat A, Vec vx){
  PetscErrorCode ierr;

  ierr = VecSet(vx, 49.0 / 9.0); CHKERRQ(ierr);

  return 0;
}
