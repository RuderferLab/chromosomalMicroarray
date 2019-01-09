#!/bin/bash
#SBATCH --mail-user=theodore.j.morley@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1    # comments allowed 
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=job_cc_pipe_slurm.out

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module load Anaconda3

# Pass your Python script to the Anaconda2 python intepreter for execution
#python match_controls.py ../data/all_record_lens_dy.csv ../data/micro_record_lens_dy.csv case_control.csv
#python translate_omop_icd.py ../data/all_cc_icd9.csv ~/PheWAS_code_translation_v1_2.txt ../files/omop_microcc_phecodes.csv ../data/all_cc_icd10.csv  ../files/ICD10CM2PHECODE.TXT
#python long_cc_micro.py ../data/gen_microarray_curated.csv case_control.csv ../files/omop_microcc_phecodes.csv ../files/long_cc_micro.csv
python summary_cc_micro.py ../files/long_cc_micro.csv ../files/omop_microcc_phecodes.csv summary_cc.csv
