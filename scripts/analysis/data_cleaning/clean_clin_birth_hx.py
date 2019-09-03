import pandas as pd
import difflib
import numpy as np
import sys
import re

PROG=re.compile('(\d+\.?\d*)(\D+)')
isnum=re.compile('\d+(\.\d+)?')

def translateToGrams(num, unit):
    num=num.strip('\n').strip('\t').strip('\r')
    unit=unit.strip('\n').strip('\t').strip('\r')
    unit=unit.replace(" ","")
    if unit=='#' or unit=='lb' or unit =='lbs':
        return float(num)*453.59237
    elif unit=='oz':
        return float(num)*28.349
    elif unit=='kg':
        return float(num)*1000
    elif unit=='g' or unit=='gr' or unit=='gm' or unit=='grams' or unit=='gms':
        return float(num)
    else:
        print(list(unit))
        print(num)
        raise Exception("Unknown input unit on birth weight found.")

def cleanBWCell(cell):
    #returns an int representing birth weight units in grams
    if '-' in cell or '/' in cell:
        return np.nan
    else:
        cell=cell.lower()
        cell=cell.replace(" ", "")
        pairs=PROG.findall(cell)
        grams=0
        for p in pairs:
            #check the unit and translate the number accordingly into grams
            grams+=float(translateToGrams(p[0], p[1]))
        return grams

def cleanTwin(cell):
    if cell=='-':
        return 0
    else:
        return 1


#Returns 1 if on term, returns 0 if not on term
def binarizeTerm(cell):
    if str(cell).lower()=='term' or str(cell)=='39':
        return 1
    else:
        return 0

def cleanTerm(cell):
    cell=str(cell).lower()
    cell=cell.strip('\n').strip('?').strip('\t').strip('\r')
    if cell=='-':
        return np.nan
    elif cell=='term':
        return 39
    elif isnum.match(cell):
        return float(cell)
    else:#if not isnum.match(cell):
        #Not a recognizable format
        print(cell)
        print('This format is non numeric, returning NA.')
        return np.nan

'''
Input:
    df: Clinical birth hx dataframe, containing a column titled BW which has nonstandardized birth weights
    Also has column titled twins
'''
def cleanBirthHX(df):
    #Observed symbols: #, lb, oz, g, gm, kg, Kg, sometimes in combination with each other. ex: 8# 10oz, 8 
    #Spacing is inconsistent.
    #Capitalization is inconsistent.
    #Significant figures are inconsistent.
    df.BW=df.BW.map(cleanBWCell)
    #df.twin=df.twin.map(cleanTwin)
    #Create binarized term/nonterm column
    df['term_binary']=df.EGA.map(binarizeTerm)
    df['EGA_clean']=df.EGA.map(cleanTerm)
    return df

#args:
##1: clinical_birth_hx
##2: output location string
def main():
    birth_hx = pd.read_csv(sys.argv[1])
    cleaned = cleanBirthHX(birth_hx)
    cleaned.to_csv(sys.argv[2],index=False)

if __name__=='__main__':
    main()
