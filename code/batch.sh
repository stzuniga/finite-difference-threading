#!/bin/bash

########################################################################
# SBATCH arguments. Learn more by running "man sbatch"  
########################################################################

## This requests 1 task. 
#SBATCH --ntasks=1

## This requests 8 cores per task.
#SBATCH --cpus-per-task=8

## This is the max run-time allowed  hr:min:sec
#SBATCH --time=00:25:00 

## This is the filename for all printed output from your code
#SBATCH --output outputHPC/fd_threading_output.txt

## This is the filename for all printed error from your code
#SBATCH --error outputHPC/fd_threading_error.txt

## The debug partition will likely be faster, but if your code takes more than
## 8 processors, you need to switch to the general partition 
#SBATCH --partition debug

## Send mail when the script ends
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=your_user_name@unm.edu 


########################################################################
########################################################################


########################################################################
# Load MPI and your custom Conda environment
########################################################################
#    To create your Conda environment for this class, you can do...
#    $ module load miniconda<tab to complete version>  
#    $ conda create --name hopper_openmpi_py3 python=3.9 numpy openmpi mpi4py scipy ipython matplotlib
#      < press y to accept installation of all new packages and the many dependencies > 

 module load singularity/3.8.5-5x3v
# source activate hopper_openmpi_py3
########################################################################
########################################################################


########################################################################
# Now, change directory into the batch system's preferred scratch directory
# (don't worry about this now)
########################################################################
cd $SLURM_SUBMIT_DIR/
########################################################################
########################################################################


########################################################################
# Now, run your program.  Just change the below filename to run a 
# different program. 
########################################################################
singularity exec fdsolver_latest-2025-06-24-bb87c9eaf86a.simg python finite_difference_threading.py
