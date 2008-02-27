ALL : ex15 shell

include ${PETSC_DIR}/bmake/common/base

nk : nk.o chkopts
	${CLINKER} -o nk nk.o ${PETSC_LIB}

ex15 : ex15.o chkopts
	${CLINKER} -o ex15 ex15.o ${PETSC_LIB}

shell : shell.o chkopts
	${CLINKER} -o shell shell.o ${PETSC_LIB}
