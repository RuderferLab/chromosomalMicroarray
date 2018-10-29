#!/bin/bash
#SBATCH --mail-user=theodore.j.morley@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1    # comments allowed 
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=job_translate_slurm.out

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module load Anaconda3

# Pass your Python script to the Anaconda2 python intepreter for execution
#python translate_icd.py ../data/icd9_micro.csv ~/PheWAS_code_translation_v1_2.txt phecodes_icd9.csv
python translate_icd.py ../data/icd9_micro.csv ~/PheWAS_code_translation_v1_2.txt phecodes.csv ../data/icd10_micro.csv ../files/ICD10CM2PHECODE.TXT phecodes.csv
