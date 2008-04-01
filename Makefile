ALL : etags stokes

PROF?=
CFLAGS+= ${PROF}
ETAGS= etags.emacs
include ${PETSC_DIR}/conf/base
#include ${PETSC_DIR}/bmake/common/base

nk : nk.o chkopts
	${CLINKER} -o nk nk.o ${PETSC_LIB}

ex15 : ex15.o chkopts
	${CLINKER} -o ex15 ex15.o ${PETSC_LIB}

shell : shell.o chkopts
	${CLINKER} -o shell shell.o ${PETSC_LIB}

cheb : cheb.o chebyshev.o chkopts
	${CLINKER} -o cheb cheb.o chebyshev.o -lfftw3 ${PETSC_LIB}

poisson : poisson.o chebyshev.o chkopts
	${CLINKER} -o poisson poisson.o chebyshev.o -lfftw3 ${PETSC_LIB}

elliptic : elliptic.o chebyshev.o chkopts
	${CLINKER} -o elliptic elliptic.o chebyshev.o ${PROF} -lfftw3 ${PETSC_LIB}

stokes : stokes.o chebyshev.o chkopts
	@-echo 'Linking stokes'
	@-${CLINKER} -o stokes stokes.o chebyshev.o ${PROF} -lfftw3 ${PETSC_LIB}

istest : istest.o chkopts
	${CLINKER} -o istest istest.o ${PETSC_LIB}

util : util.o chkopts
	${CLINKER} -o util util.o ${PETSC_LIB}

etags : chebyshev.c stokes.C util.C
	${ETAGS} $^
