
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
