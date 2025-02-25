#!/bin/bash
PROG=../../../build/src/main/run_ocean_loca
mpirun -np 4 ${PROG}
