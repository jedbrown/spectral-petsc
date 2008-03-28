#ifdef MAIN
#include <stdio.h>
#include <petscvec.h>
#endif

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
  int shift(int j, int s) const {
    const int is = ind[j] + s;
    if (is < 0 || is >= dim[j]) return -1;
    return i + s * stride[j];
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

#undef __FUNCT__
#define __FUNCT__ "polyInterp"
PetscErrorCode polyInterp(const PetscInt n, const PetscReal *x, PetscScalar *w, const PetscReal x0, const PetscReal x1, PetscScalar *f0, PetscScalar *f1)
{
  PetscScalar *tmp;
  PetscInt o, e;
  PetscReal y;

  PetscFunctionBegin;
  for (int di=1; di < n; di++) { // offset (column of table)
    o = di % 2; e = (o+1) % 2;
    for (int i=0; i < n-di; i++) { // nodes
      w[i*4+2*o]   = ((x0-x[i+di])*w[i*4+2*e  ] + (x[i]-x0)*w[(i+1)*4+2*e  ]) / (x[i] - x[i+di]);
      w[i*4+2*o+1] = ((x1-x[i+di])*w[i*4+2*e+1] + (x[i]-x1)*w[(i+1)*4+2*e+1]) / (x[i] - x[i+di]);
    }
  }
  *f0 = w[2*o];
  *f1 = w[2*o+1];
  PetscFunctionReturn(0);
}

#ifdef MAIN
#define N 20
PetscScalar func(PetscReal x) {
  //return pow(x,6) + 3.1*pow(x,4) + 2.7*pow(x,3);
  return cos(x);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscReal x[N], w[N*4], x0, x1, f0, f1;
  PetscErrorCode ierr;

  for (int order = 2; order < N; order++) {
    for (int i=0; i < order; i++) {
      x[i] = 1.0 + 1.0 * i;
      w[i*4] = func(x[i]);
      w[i*4+1] = w[i*4];
    }
    x0 = 1.43; x1 = 3.1;
    ierr = polyInterp(order, x, w, x0, x1, &f0, &f1);CHKERRQ(ierr);
    printf("[%d], %f ~ %f    %f ~ %f\n", order, f0, func(x0), f1, func(x1));
  }
  return 0;
}

#endif
