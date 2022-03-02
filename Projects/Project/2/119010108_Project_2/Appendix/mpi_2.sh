#!/bin/bash
#SBATCH --account=csc4005
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4




echo "mainmode: " && /bin/hostname
xvfb-run -a mpirun -n  /pvfsmnt/119010108/Project_2/MPI_Version/build/csc4005_imgui

