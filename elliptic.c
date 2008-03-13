static char help[] = "A nonlinear elliptic equation by Chebyshev differentiation.\n";

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
#include <petscsnes.h>
#include <stdbool.h>

typedef struct {
  int d, *dim;
  Mat Dx, Dy;
  Vec w0, w1, w2, w3;
  Vec nu;
  Vec dirichlet, dirichlet0;
  VecScatter scatterLG, scatterGL, scatterLD, scatterDL;
} MatElliptic;

typedef struct {
  PetscInt m, n;
  Mat Dx, Dy;
  Vec nu;
} AppCtx;

typedef enum { BDY_DIRICHLET, BDY_NEUMANN } BdyType;

typedef struct {
  BdyType type;
  PetscScalar value;
} BdyCond;

typedef PetscErrorCode(*BdyFunc)(int, double *, BdyCond *);

extern PetscErrorCode MatCreate_Elliptic(MPI_Comm comm, int d, int *dim, unsigned flag, BdyFunc bf, Vec *vG, Mat *A);
extern PetscErrorCode MatMult_Elliptic(Mat, Vec, Vec);
extern PetscErrorCode MatDestroy_Elliptic(Mat);
extern PetscErrorCode SetupBC(MPI_Comm comm, BdyFunc bf, Vec *vGlob, MatElliptic *c);
extern PetscErrorCode LiftDirichlet_Elliptic(Mat A, Vec x, Vec b);
extern PetscErrorCode DirichletBdy(int d, double *x, BdyCond *bc);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, AppCtx *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x, b, b0, u, u2;
  Mat            A, P;
  KSP            ksp;
  //SNES           snes;
  PetscReal      norm;
  PetscInt       m = 5, n = 7, p=1, dd, its, exact=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,(char *)0,help); CHKERRQ(ierr);

  ierr = fftw_import_system_wisdom(); CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-exact",&exact,PETSC_NULL);CHKERRQ(ierr);

  if (n == 1) dd = 1;
  else if (p == 1) dd = 2;
  else dd = 3;
  int dim[] = { m, n, p };

  ierr = MatCreate_Elliptic(PETSC_COMM_WORLD, dd, dim, FFTW_ESTIMATE, DirichletBdy, &u, &A); CHKERRQ(ierr);

  if (false) {
    MatElliptic *c;
    ierr = VecView(u, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
    ierr = VecSet(u, 5.0); CHKERRQ(ierr);
    ierr = VecSet(c->dirichlet, 10.0); CHKERRQ(ierr);
    ierr = VecScatterBegin(c->scatterGL, u, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(c->scatterGL, u, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterBegin(c->scatterDL, c->dirichlet, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(c->scatterDL, c->dirichlet, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecView(c->w0, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }

  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &b0);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &u2);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &x);CHKERRQ(ierr);

  // 3-D solution function
  MatElliptic *c;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  double *w0, *w1, *dir;
  ierr = VecGetArray(c->w0, &w0); CHKERRQ(ierr);
  ierr = VecGetArray(c->w1, &w1); CHKERRQ(ierr);
  for (int i=0; i < m; i++) {
    double x = (m==1) ? 0 : cos (i * PI / (m-1));
    for (int j=0; j < n; j++) {
      double y = (n==1) ? 0 : cos (j * PI / (n-1));
      for (int k=0; k < p; k++) {
        // double z = (p==1) ? 0 : cos (k * PI / (p-1));
        int ix = (i*n + j) * p + k;
        switch (exact) {
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
  ierr = VecRestoreArray(c->w0, &w0); CHKERRQ(ierr);
  ierr = VecRestoreArray(c->w1, &w1); CHKERRQ(ierr);
  // ierr = VecView(c->w0, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  ierr = VecScatterBegin(c->scatterLG, c->w0, u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w0, u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->w1, u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w1, u2, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLD, c->w0, c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLD, c->w0, c->dirichlet, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // ierr = VecView(u, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
  // ierr = VecView(u2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");

  ierr = VecSet(x, 0.0); CHKERRQ(ierr);
  ierr = LiftDirichlet_Elliptic(A, x, b0); CHKERRQ(ierr);
  ierr = VecCopy(u2, b); CHKERRQ(ierr);
  ierr = VecAXPY(b, -1.0, b0); CHKERRQ(ierr);

#if CHECK_EXACT
  ierr = MatMult(A, u, x); CHKERRQ(ierr);
  ierr = VecAXPY(x, 1.0, b0); CHKERRQ(ierr);
  //ierr = VecCopy(b, x); CHKERRQ(ierr);
  ierr = VecAXPY(x, -1.0, u2);CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A\n",norm);CHKERRQ(ierr);
#endif

#if SOLVE
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

  ierr = VecAXPY(x, -1.0, u); CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_INFINITY, &norm); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm of error %A iterations %D\n", norm, its); CHKERRQ(ierr);
#endif

  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);

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
  ierr = PetscMalloc(d*sizeof(PetscInt), &c->dim); CHKERRQ(ierr);
  c->d = d;
  for (int i=0; i<d; i++) c->dim[i] = dim[i];
  PetscInt m = 1;
  for (int i=0; i<d; i++) m *= dim[i];

  ierr = VecCreateSeq(comm, m, &(c->w0)); CHKERRQ(ierr);
  ierr = VecDuplicate(c->w0, &(c->w1)); CHKERRQ(ierr);
  ierr = VecDuplicate(c->w0, &(c->w2)); CHKERRQ(ierr);
  ierr = VecDuplicate(c->w0, &(c->w3)); CHKERRQ(ierr);
  ierr = VecDuplicate(c->w0, &(c->nu)); CHKERRQ(ierr);
  ierr = VecZeroEntries(c->nu); CHKERRQ(ierr);

  ierr = MatCreateCheb(comm, d, 0, dim, flag, c->w0, c->w1, &c->Dx); CHKERRQ(ierr);
  ierr = MatCreateCheb(comm, d, 1, dim, flag, c->w0, c->w1, &c->Dy); CHKERRQ(ierr);

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
PetscErrorCode MatMult_Elliptic(Mat A, Vec u, Vec v) {
  PetscErrorCode ierr;
  MatElliptic *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, u, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, u, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, c->dirichlet0, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, c->dirichlet0, c->w0, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  /* ierr = MatMultAdd(c->Dx, c->w0, c->nu, c->w1); CHKERRQ(ierr); */
  /* ierr = MatMultAdd(c->Dy, c->w0, c->nu, c->w2); CHKERRQ(ierr); */
  ierr = MatMult(c->Dx, c->w0, c->w1); CHKERRQ(ierr);
  ierr = MatMult(c->Dy, c->w0, c->w2); CHKERRQ(ierr);
  ierr = MatMult(c->Dx, c->w1, c->w3); CHKERRQ(ierr);
  ierr = MatMult(c->Dy, c->w2, c->w0); CHKERRQ(ierr);
  ierr = VecAXPBY(c->w0, -1.0, -1.0, c->w3); CHKERRQ(ierr);
  //ierr = MatMultAdd(c->Dy, c->w2, c->w3, c->w0); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterLG, c->w0, v, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterLG, c->w0, v, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes, Vec u, Vec rhs, AppCtx *c) {
  //  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Elliptic"
PetscErrorCode MatDestroy_Elliptic (Mat A) {
  PetscErrorCode ierr;
  MatElliptic *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = MatDestroy(c->Dx); CHKERRQ(ierr);
  ierr = MatDestroy(c->Dy); CHKERRQ(ierr);
  ierr = VecDestroy(c->w0); CHKERRQ(ierr);
  ierr = VecDestroy(c->w1); CHKERRQ(ierr);
  ierr = VecDestroy(c->w2); CHKERRQ(ierr);
  ierr = VecDestroy(c->w3); CHKERRQ(ierr);
  ierr = VecDestroy(c->nu); CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet); CHKERRQ(ierr);
  ierr = VecDestroy(c->dirichlet0); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterGL); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterDL); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLG); CHKERRQ(ierr);
  ierr = VecScatterDestroy(c->scatterLD); CHKERRQ(ierr);
  ierr = PetscFree(c->dim); CHKERRQ(ierr);
  ierr = PetscFree(c);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupBC"
PetscErrorCode SetupBC(MPI_Comm comm, BdyFunc bf, Vec *vGlob, MatElliptic *c) {
  PetscErrorCode ierr;
  IS isG, isD;
  PetscInt *ixG, *ixD;
  PetscScalar *uD;
  Vec vL = c->w0; // A prototype local vector
  Vec vG, vD;
  const PetscInt m=c->dim[0], n=c->dim[1];

  PetscFunctionBegin;
  ierr = PetscMalloc2(m*n, PetscInt, &ixG, 2*(m+n), PetscInt, &ixD); CHKERRQ(ierr);
  ierr = VecGetArray(c->w1, &uD); CHKERRQ(ierr);
  PetscInt l = 0, g = 0, d = 0;
  for (int i=0; i < m; i++) {
    for (int j=0; j < n; j++) {
      if (i==0 || i==m-1 || j==0 || j==n-1) {
        double x[2] = { cos(i*PI / (m-1)), cos(j*PI / (n-1)) };
        BdyCond bc;
        ierr = bf(2, x, &bc); CHKERRQ(ierr);
        if (bc.type == BDY_DIRICHLET) {
          uD[d] = bc.value;
          ixD[d++] = l++;
        } else { SETERRQ(1, "Neumann not implemented."); }
      } else {
        ixG[g++] = l++;
      }
    }
  }
  ierr = VecCreateSeq(comm, g, &vG); CHKERRQ(ierr);
  ierr = VecCreateSeq(comm, d, &vD); CHKERRQ(ierr);
  {
    PetscScalar *v;
    ierr = VecGetArray(vD, &v); CHKERRQ(ierr);
    for (int i=0; i<d; i++) v[i] = uD[i];
    ierr = VecRestoreArray(vD, &v); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(c->w1, &uD); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, g, ixG, &isG); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, d, ixD, &isD); CHKERRQ(ierr);
  if (false) {
    PetscTruth flag;
    ierr = ISView(isD, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
    ierr = VecView(vD, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
    ierr = ISView(isG, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
    ierr = VecView(vG, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr); printf("\n");
    ierr = VecView(vL, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }
  ierr = VecScatterCreate(vD, PETSC_NULL, vL, isD, &c->scatterDL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vG, PETSC_NULL, vL, isG, &c->scatterGL); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, isD, vD, PETSC_NULL, &c->scatterLD); CHKERRQ(ierr);
  ierr = VecScatterCreate(vL, isG, vG, PETSC_NULL, &c->scatterLG); CHKERRQ(ierr);

#if 0
  ierr = VecSet(vL, 0.0); CHKERRQ(ierr);
  ierr = VecSet(vG, 1.0); CHKERRQ(ierr);
  ierr = VecSet(vD, 2.0); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterGL, vG, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterGL, vG, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(c->scatterDL, vD, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(c->scatterDL, vD, vL, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecView(vL, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
#endif

  ierr = PetscFree2(ixG, ixD); CHKERRQ(ierr);
  ierr = ISDestroy(isG); CHKERRQ(ierr);
  ierr = ISDestroy(isD); CHKERRQ(ierr);
  *vGlob = vG;
  c->dirichlet = vD;
  ierr = VecDuplicate(vD, &c->dirichlet0); CHKERRQ(ierr);
  ierr = VecZeroEntries(c->dirichlet0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DirichletBdy"
PetscErrorCode DirichletBdy(int dim, double *x, BdyCond *bc) {
  PetscErrorCode ierr;

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
