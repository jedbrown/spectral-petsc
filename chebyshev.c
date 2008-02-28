#include "chebyshev.h"

#undef __FUNCT__
#define __FUNCT__ "MatCreateChebD1"
PetscErrorCode MatCreateChebD1(MPI_Comm comm, Vec vx, Vec vy, unsigned flag, Mat *A) {
  PetscErrorCode ierr;
  ChebD1Ctx *c;
  int n;
  double *x, *y;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(ChebD1Ctx), &c); CHKERRQ(ierr);
  ierr = VecGetSize(vx, &(c->n)); CHKERRQ(ierr);
  ierr = VecGetSize(vy, &n); CHKERRQ(ierr);
  if (c->n != n || n < 2) SETERRQ1(PETSC_ERR_USER, "n = %d but must be >= 2", n);

  c->work = fftw_malloc(n * sizeof(double));
  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);
  c->p_forward = fftw_plan_r2r_1d(n, x, c->work, FFTW_REDFT00, flag);
  c->p_backward = fftw_plan_r2r_1d(n-2, c->work + 1, y + 1, FFTW_RODFT00, flag);
  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  ierr = MatCreateShell(comm, n, n, n, n, c, A); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))ChebD1Mult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))ChebD1Destroy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ChebD1Mult"
PetscErrorCode ChebD1Mult(Mat A, Vec vx, Vec vy) {
  PetscErrorCode ierr;
  double *x, *y;
  ChebD1Ctx *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void *)&c); CHKERRQ(ierr);
  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  int n = c->n - 1; // Gauss-Lobatto-Chebyshev points are numbered from [0..n]

  fftw_execute_r2r(c->p_forward, x, c->work);
  for (int i = 1; i < n; i++) c->work[i] *= (double)i;

  fftw_execute_r2r(c->p_backward, c->work + 1, y + 1);

  double N = (double)n;
  double pin = PI / N;
  y[0] = 0.0;
  y[n] = 0.0;
  double s = 1.0;
  for (int i = 1; i < n; i++) {
    double I = (double)i;
    y[i] /= 2.0 * n * sqrt(1.0 - PetscSqr(cos(I * pin)));
    y[0] += I * c->work[i];
    y[n] += s * I * c->work[i];
    s = -s;
  }
  y[0] = 0.5 * c->work[n] * N + y[0] / n;
  y[n] = y[n] / N + 0.5 * s * N * c->work[n];

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ChebD1Destroy"
PetscErrorCode ChebD1Destroy (Mat A) {
  PetscErrorCode ierr;
  ChebD1Ctx *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void *)&c); CHKERRQ(ierr);
  fftw_destroy_plan(c->p_forward);
  fftw_destroy_plan(c->p_backward);
  fftw_free(c->work);
  PetscFunctionReturn(0);
}

