#!/bin/bash

# ---------- How to use Scitas ----------
# 1. Connect to the cluster: ssh -X <username>@<cluster>.epfl.ch in the terminal (-X if you want GPU)
# 2. Upload the files to the cluster:[sabatier@izar1 ~]$ scp -r <root_of_the_src> <GASPAR>@izar.epfl.ch:<root_of_the_dst>
# exemple: scp -r <root_of_the_src> sabatier@izar.epfl.ch:/home/sabatier/files
# 3. Create a module (is a utility that allows multiple, often incompatible, tools and libraries to co-exist on a system)
# exemple: module load gcc python/3.10.4
# 4. Create a virtual environment: virtualenv --system-site-packages ~/venvs/<env_name>
# exemple: virtualenv --system-site-packages ~/venvs/env_deeplearning
# THE REST CAN BE DONE IN THE SCRIPT:
# 5. Activate the virtual environment: source ~/venvs/<env_name>/bin/activate
# exemple: source ~/venvs/env_deeplearning/bin/activate
# 6. Install the necessary packages written in a .txt file in the directory you want to run the file(scratch/izar): pip install -r requirements.txt
# 7. write a script in a bash file
# 8. submit the script to the cluster: sbatch <script>.sh

# ---------- Commands to set up SLURM environment----------
# Example of a submission script for the EPFL SCITAS cluster:

# Directory where the job will be executed and where the output files will be written:
# Use /home to have back up files (100GB), else use /scratch
#SBATCH --chdir /scratch/izar/sabatier/deep_learning

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
#SBATCH --output /scratch/izar/sabatier/deep_learning/output_file.txt

# Output file for the errors:
#SBATCH --error /scratch/izar/sabatier/deep_learning/error_file.txt

# Name of the account to have more resources:
#SBATCH --account ee-559

# Name of the course to have a priority:
#SBATCH --qos ee-559

# ---------- Job to be executed ----------
source ~/venvs/env_deeplearning/bin/activate
pip install -r ./requirements.txt
echo "start running file"
python ./folder/file.py

# ---------- Check the job status ----------
# ATTENTION: keep track of the job ID (2088720)
# Sjob <JOB_ID>: check the status of a job
# Squeue: list all your jobs and their status (if none it means the job is running or finished)
# squeue -A ee-559: list all the jobs of the course
# scancel <JOB_ID>: cancel a job
# scancel -u <username>: cancel all your jobs
# To download the output files: scp -r sabatier@izar.epfl.ch:/scratch/izar/sabatier/deep_learning/marked10 <root_of_the_dst>
# note: the output files are in the directory where the job was executed
# note: it is easier to download or upload from a local terminal on your computer,
# move to the direction where you want to download the files and use the scp command with a "." instead of the <root_of_the_dst>

