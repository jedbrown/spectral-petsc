#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

// #include <math.h>
#include "fftw3.h"
#include "petscmat.h"
// #include "petscerror.h"

#define PI 3.14159265358979323846

typedef struct {
  int n;
  fftw_plan p_forward, p_backward;
  double *work;
} ChebD1Ctx;

PetscErrorCode MatCreateChebD1(MPI_Comm comm, Vec vx, Vec vy, unsigned flag, Mat *A);
PetscErrorCode ChebD1Mult(Mat A, Vec vx, Vec vy);
PetscErrorCode ChebD1Destroy (Mat A);

#endif // CHEBYSHEV_H
