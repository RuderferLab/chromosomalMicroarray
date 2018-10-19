#!/bin/bash
#SBATCH --mail-user=theodore.j.morley@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1    # comments allowed 
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=18G
#SBATCH --output=job_micro_slurm.out

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module load Anaconda3

# Pass your Python script to the Anaconda2 python intepreter for execution
python micro.py gen_microarray.csv matching_id_micro.csv ~/PheWAS_code_translation_v1_2.txt micro_grid_dob.csv full_micro_duplicates_dropped.csv
