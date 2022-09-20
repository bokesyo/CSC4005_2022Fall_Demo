#!/bin/bash
#SBATCH --job-name=parallel_job_test # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=4                   # number of processes = 4
#SBATCH --cpus-per-task=4            # Number of CPU cores per process
#SBATCH --mem=600mb                  # Total memory limit
#SBATCH --time=00:05:00              # Time limit hrs:min:sec
#SBATCH --partition=Debug            # Partition name: Project or Debug (Debug is default)

mpirun -np 4 ./xxxxx
