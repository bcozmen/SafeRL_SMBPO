#!/bin/bash


#SBATCH -J rl1		# Job Name
#SBATCH --nodes=1 		# Anzahl Knoten N
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G 

##Max Walltime vorgeben:
#SBATCH --time=2-00:05:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.oezmen@campus.tu-berlin.de

# benötigte SW / Bibliotheken laden
module load python
python3 main.py


