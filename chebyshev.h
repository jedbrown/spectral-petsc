/*
#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
#define PETSC_EXTERN_CXX_BEGIN extern "C" {
#define PETSC_EXTERN_CXX_END  }
#else
#define PETSC_EXTERN_CXX_BEGIN
#define PETSC_EXTERN_CXX_END
#endif
*/
#include <petscversion.h>
#include <petscsys.h>
/*
#ifndef PETSC_EXTERN_CXX_BEGIN
# define PETSC_EXTERN_CXX_BEGIN
# define PETSC_EXTERN_CXX_END
#endif
*/

#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

// #include <math.h>
#include <fftw3.h>
#include <petscmat.h>


//PETSC_EXTERN_CXX_BEGIN
extern "C" {

#define PI 3.14159265358979323846

typedef struct {
  int n;
  fftw_plan p_forward, p_backward;
  double *work;
} ChebD1Ctx;

typedef struct {
  int rank, tr;
  fftw_iodim tdim;
  fftw_iodim *dim;
  fftw_plan p_forward, p_backward;
  double *work;
} ChebCtx;


PetscErrorCode MatCreateChebD1(MPI_Comm comm, Vec vx, Vec vy, unsigned int flag, Mat *A);
PetscErrorCode ChebD1Mult(Mat A, Vec vx, Vec vy);
PetscErrorCode ChebD1Destroy (Mat A);

PetscErrorCode MatCreateCheb(MPI_Comm comm, int rank, int tr, int *dims, unsigned int flag,
                              Vec vx, Vec vy, Mat *A);
PetscErrorCode ChebMult(Mat A, Vec vx, Vec vy);
PetscErrorCode ChebDestroy (Mat A);


//PETSC_EXTERN_CXX_END
}
#endif // CHEBYSHEV_H
