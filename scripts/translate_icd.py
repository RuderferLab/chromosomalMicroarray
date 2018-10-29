import sys
import pandas as pd

def translate_icd9():
    #GRID, ENTRY_DATE, CODE
    icd = open(sys.argv[1], 'r')
    phecode_table = pd.read_table(sys.argv[2])
    out = open(sys.argv[3], 'w')
    for line in icd:
        line = line.strip("\n")
        GRID, date, code = line.split(',')
        if code in phecode_table['icd9'].values:
            row = phecode_table.loc[phecode_table['icd9']==code]
            out.write(GRID+','+date+','+str(row['phewas_code'].values[0])+'\n')
    out.close()
    icd.close()

def translate_icd10():
    #Take in file with icd10 codes, translation key for phecodes, and existing file with phecodes (append to this file)
    icd = open(sys.argv[4], 'r')
    #phecode table: ICD10CM,PHECODE
    phecode_table = pd.read_csv(sys.argv[5])
    out = open(sys.argv[6], 'a')
    for line in icd:
        line = line.strip("\n")
        GRID, date, code = line.split(',')
        if code in phecode_table['ICD10CM'].values:
            row = phecode_table.loc[phecode_table['ICD10CM']==code]
            out.write(GRID+','+date+','+str(row['PHECODE'].values[0])+'\n')
    out.close()
    icd.close()

if __name__ == "__main__":
    translate_icd9()
    translate_icd10()
