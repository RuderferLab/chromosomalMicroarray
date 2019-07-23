import pandas as pd
import numpy as np
import sys
import re

PROG=re.compile('(\d+)(\D+)')

if __name__=='__main__':
    main()

def translateToGrams(num, unit):
    if unit=='#' or unit=='lb' or unit =='lbs':
        return num*453.59237
    elif unit=='kg':
        return num*1000
    elif unit=='g' or unit=='gr' or unit=='gm':
        return num
    else:
        raise Exception("Unknown input unit on birth weight found.")

def cleanCell(cell):
    #returns an int representing birth weight units in grams
    if cell=='-':
        return np.nan
    else:
        cell=cell.lower()
        cell=cell.replace(" ", "")
        pairs=prog.findall(cell)
        grams=0
        for p in pairs:
            #check the unit and translate the number accordingly into grams
            grams+=translateToGrams(p[0], p[1])
        return grams

'''
Input:
    df: Clinical birth hx dataframe, containing a column titled BW which has nonstandardized birth weights
'''
def cleanBirthWeight(df):
    #Observed symbols: #, lb, oz, g, gm, kg, Kg, sometimes in combination with each other. ex: 8# 10oz, 8 
    #Spacing is inconsistent.
    #Capitalization is inconsistent.
    #Significant figures are inconsistent.
    df.BW=df.BW.map(cleanCell)

#args:
##1: clinical_birth_hx
def main():
    birth_hx = pd.read_csv(sys.argv[1])
