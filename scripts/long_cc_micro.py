import sys
import datetime
from pandas import read_csv, read_table, DataFrame


def uc_filter(d):
    return d[d.result_text.str.contains("unclear")==False]

def dconvert(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d")

def dconvert_s(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S.0")

def dextract(d):
    return d.days

def year_calc(GRID, d_set):
    return (d_set[GRID][1]-d_set[GRID][0]).days/365

def uc_mark(x):
    if "unclear" in x.lower():
        return 1
    else:
        return 0

#microarray demog_controls phecodes out
if __name__=="__main__":
    #df = microarray results (curated)
    df = read_csv(sys.argv[1])
    #Filter out all duplicates
    df = df.drop_duplicates(subset="GRID")
    #Mark all unclear results
    df['unclear'] = df['result_text'].apply(uc_mark)
    #Merge with demographic data and controls
    demog = read_csv(sys.argv[2])
    df = df.merge(demog, on="GRID", how="outer")
    #load in phecodes (9 and 10)
    phecodes = read_csv(sys.argv[3], dtype=str)#, names=['GRID', 'CODE', 'ENTRY_DATE'])
    phecodes.columns = ['GRID', 'CODE']#, 'ENTRY_DATE']
    phecodes_group = phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    df = df.merge(phecodes_group, on="GRID", how="left")
    df[phecodes.CODE.unique()] = df[phecodes.CODE.unique()].fillna(0).astype(int)
    print(df.head())
    df.to_csv(sys.argv[4], index=False)
