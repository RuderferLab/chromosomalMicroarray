#!/bin/bash
#SBATCH --mail-user=theodore.j.morley@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1    # comments allowed 
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=job_long_micro_slurm.out

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module load Anaconda3

# Pass your Python script to the Anaconda2 python intepreter for execution
python long_micro.py ../data/gen_microarray.csv ../data/matching_id_micro.csv ~/PheWAS_code_translation_v1_2.txt ../data/dob.csv ../data/total_years.csv long_micro.csv
