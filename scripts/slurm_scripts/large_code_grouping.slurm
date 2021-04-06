#!/bin/bash
#SBATCH --mail-user=theodore.j.morley@vumc.org
#SBATCH --mail-type=ALL
#SBATCH --nodes=1    # comments allowed 
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=100GB
#SBATCH --output=code_group.out
#SBATCH --account=vgi

module load Anaconda3

python ../group_large_codes.py /data/ruderferlab/projects/sd/data/icd.csv ICD10CM ../../files/grouped_icd10_size.csv
