#include "chebyshev.h"
#include <stdbool.h>

void perform_carry(ChebCtx *c, int *ind, int *offset, bool *done);

#undef __FUNCT__
#define __FUNCT__ "MatCreateChebD1"
PetscErrorCode MatCreateChebD1(MPI_Comm comm, Vec vx, Vec vy, unsigned int flag, Mat *A) {
  PetscErrorCode ierr;
  ChebD1Ctx *c;
  int n;
  double *x, *y;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(ChebD1Ctx), &c); CHKERRQ(ierr);
  ierr = VecGetSize(vx, &(c->n)); CHKERRQ(ierr);
  ierr = VecGetSize(vy, &n); CHKERRQ(ierr);
  if (c->n != n || n < 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "n = %d but must be >= 2", n);

  c->work = (double *)fftw_malloc(n * sizeof(double));
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
  PetscInt i;
  double *x, *y;
  ChebD1Ctx *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  int n = c->n - 1; // Gauss-Lobatto-Chebyshev points are numbered from [0..n]

  fftw_execute_r2r(c->p_forward, x, c->work);
  for (i = 1; i < n; i++) c->work[i] *= (double)i;

  fftw_execute_r2r(c->p_backward, c->work + 1, y + 1);

  double N = (double)n;
  double pin = PI / N;
  y[0] = 0.0;
  y[n] = 0.0;
  double s = 1.0;
  for (i = 1; i < n; i++) {
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
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  fftw_destroy_plan(c->p_forward);
  fftw_destroy_plan(c->p_backward);
  fftw_free(c->work);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateCheb"
PetscErrorCode MatCreateCheb(MPI_Comm comm, int rank, int tr, int *dim, unsigned int flag,
                             Vec vx, Vec vy, Mat *A) {
  PetscErrorCode ierr;
  ChebCtx *c;
  int n, r, ri, stride;
  double *x, *y;

  PetscFunctionBegin;
  ierr = VecGetSize(vx, &n); CHKERRQ(ierr);
  if (n < 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "n = %d but must be >= 2", n);

  ierr = PetscMalloc(sizeof(ChebCtx), &c); CHKERRQ(ierr);
  ierr = PetscMalloc((rank - 1) * sizeof(fftw_iodim), &(c->dim)); CHKERRQ(ierr);
  c->work = (double *)fftw_malloc(n * sizeof(double));
  c->rank = rank;
  c->tr = tr;

  if (!(0 <= tr && tr < rank)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "tdim out of range");
  stride = 1;
  for (r = rank-1, ri=rank-2; r >= 0; r--) {
    fftw_iodim *iod;
    if (r == tr) {
      iod = &(c->tdim);
    } else {
      iod = &(c->dim[ri]);
      ri--;
    }
    iod->n = dim[r];
    iod->is = stride;
    iod->os = stride;
    stride *= dim[r];
  }

  if (n != stride) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER, "dimensions do not agree: n = %d but stride = %d", n, stride);

  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);
  const fftw_r2r_kind redft00 = FFTW_REDFT00, rodft00 = FFTW_RODFT00;
  c->p_forward = fftw_plan_guru_r2r(1, &(c->tdim), c->rank-1, c->dim, x, c->work, &redft00, flag | FFTW_PRESERVE_INPUT);
  fftw_iodim tdim1 = c->tdim; tdim1.n -= 2; // we need a modified tdim to come back.
  c->p_backward = fftw_plan_guru_r2r(1, &tdim1, c->rank-1, c->dim, c->work + tdim1.os, y + tdim1.is, &rodft00, flag | FFTW_DESTROY_INPUT);
  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  ierr = MatCreateShell(comm, n, n, n, n, c, A); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))ChebMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))ChebDestroy); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ChebMult"
PetscErrorCode ChebMult(Mat A, Vec vx, Vec vy) {
  PetscErrorCode ierr;
  double *x, *y;
  bool done;
  int i;
  ChebCtx *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  ierr = VecGetArray(vx, &x); CHKERRQ(ierr);
  ierr = VecGetArray(vy, &y); CHKERRQ(ierr);

  int n = c->tdim.n - 1; // Gauss-Lobatto-Chebyshev points are numbered from [0..n]
  int ind[c->rank-1]; //ierr = PetscMalloc(c->rank * sizeof(int), &ind); CHKERRQ(ierr);

  fftw_execute_r2r(c->p_forward, x, c->work);

  double N = (double)n;
  ierr = PetscMemzero(ind, (c->rank - 1) * sizeof(int)); CHKERRQ(ierr);
  int offset = 0;
  for (done = false; !done; ) {
    int ix0 = offset;
    int ixn = offset + n * c->tdim.is;
    y[ix0] = 0.0;
    y[ixn] = 0.0;
    double s = 1.0;
    for (i = 1; i < n; i++) {
      int ix = offset + i * c->tdim.is;
      double I = (double)i;
      c->work[ix] *= I;
      y[ix0] += I * c->work[ix];
      y[ixn] += s * I * c->work[ix];
      s = -s;
    }
    y[ix0] = 0.5 * c->work[ixn] * N + y[ix0] / n;
    y[ixn] = y[ixn] / N + 0.5 * s * N * c->work[ixn];
    perform_carry(c, ind, &offset, &done);
  }

  fftw_execute_r2r(c->p_backward, c->work + c->tdim.os, y + c->tdim.is);

  double pin = PI / N;
  ierr = PetscMemzero(ind, (c->rank - 1) * sizeof(int)); CHKERRQ(ierr);
  offset = 0;
  for (done = false; !done; ) {
    for (i = 1; i < n; i++) {
      int ix = offset + i * c->tdim.is;
      double I = (double)i;
      y[ix] /= 2 * n * sqrt(1.0 - PetscSqr(cos(I * pin)));
    }
    perform_carry(c, ind, &offset, &done);
  }

  ierr = VecRestoreArray(vx, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(vy, &y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


void perform_carry(ChebCtx *c, int *ind, int *offset, bool *done) {
  int carry, i;

  *offset = 0;
  carry = 1;
  for (i = c->rank-2; i >= 0; i--) {
    ind[i] += carry;
    if (ind[i] < c->dim[i].n) {
      carry = 0;
    } else {
      ind[i] = 0;
      carry = 1;
    }
    *offset += ind[i] * c->dim[i].is;
  }
  *done = (carry == 1);
}


#undef __FUNCT__
#define __FUNCT__ "ChebDestroy"
PetscErrorCode ChebDestroy (Mat A) {
  PetscErrorCode ierr;
  ChebCtx *c;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A, (void **)&c); CHKERRQ(ierr);
  fftw_destroy_plan(c->p_forward);
  fftw_destroy_plan(c->p_backward);
  fftw_free(c->work);
  ierr = PetscFree(c->dim); CHKERRQ(ierr);
  ierr = PetscFree(c); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

