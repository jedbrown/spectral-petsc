ALL : ex15 shell
CFLAGS= -std=c99

include ${PETSC_DIR}/conf/base

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
