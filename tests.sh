#!/bin/sh

inc=$1

test_range () {
    cos_scale=$1;
    echo 'cos_scale = ' $cos_scale
    for n in $(seq $2 $3 $4); do
	printf "%3d: " $n
	./elliptic -dims $n,$n -exact 0 -cos_scale $cos_scale -gamma 4 -ksp_rtol 1e-12 -snes_rtol 1e-12 | grep 'Norm of error'
    done
}

test_range 3 4 $inc 44
test_range "2.8" 4 $inc 44
