#!/bin/bash

# ---------- How to use Scitas ----------
# 1. Connect to the cluster: ssh -X <username>@<cluster>.epfl.ch in the terminal (-X if you want GPU)
# 2. Upload the files to the cluster:[sabatier@izar1 ~]$ scp -r <root_of_the_src> sabatier@izar.epfl.ch:<root_of_the_dst>
# 3. write a script in a bash file
# 4. submit the script to the cluster: sbatch <script>.sh

# ---------- Commands to set up SLURM environment----------
# Example of a submission script for the EPFL SCITAS cluster:

# Directory where the job will be executed and where the output files will be written:
# Use /home to have back up files (100GB), else use /scratch
#SBATCH --chdir /scratch/izar/sabatier

# Number of MPI tasks per node (default is 1):
#SBATCH --ntasks 1

# Number of cores per task:
#SBATCH --cpus-per-task 8

# Number of RAM needed for the job (in GB, between 1G and 128G):
#SBATCH --mem 32G

# Maximum run time of the job (format: D-HH:MM:SS):
#SBATCH --time 1:00:00

# Number of GPUs needed for the job:
#SBATCH --gres gpu:1

# Output file for the scitas messages:
#SBATCH --output /scratch/izar/sabatier/output_marked10.txt

# Output file for the errors:
#SBATCH --error /scratch/izar/sabatier/error_marked10.txt

# Name of the account to have more resources:
#SBATCH --account ee-559

# Name of the course to have a priority:
#SBATCH --qos ee-559

# ---------- Job to be executed ----------
source ~/venvs/deep_learning_py-3.11.8/bin/activate
python /home/sabatier/files/main.py

# ---------- Check the job status ----------
# ATTENTION: keep track of the job ID (2087187)
# Sjob <JOB_ID>: check the status of a job
# Squeue: list all your jobs and their status (if none it means the job is running or finished)
# squeue -A ee-559: list all the jobs of the course
# scancel <JOB_ID>: cancel a job
# scancel -u <username>: cancel all your jobs
