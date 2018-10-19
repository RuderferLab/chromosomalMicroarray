import sys
import datetime
from pandas import read_csv, read_table, DataFrame


def uc_filter(d):
    return d[d.result_text.str.contains("unclear")==False]

def dconvert(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d")

def dextract(d):
    return d.days

def year_calc(GRID, d_set):
    return (d_set[GRID][1]-d_set[GRID][0]).days/365

if __name__=="__main__":
    df = read_csv(sys.argv[1], quotechar="\"", escapechar="\\")
    #Filter out all with unclear in result_text
    df = uc_filter(df)
    #print("','".join(df.GRID))
    #Load in ICD for the given GRIDS
    icd = read_csv(sys.argv[2])
    #Load in Phecode translation
    phecode_table = read_table(sys.argv[3])
    #Load in dob
    dobs = read_csv(sys.argv[4])
    #match dobs with grids in results
    df = df.merge(dobs, on="GRID", how="inner")
    #Create age column based off of dob and ENTRY_DATE 
    df.ENTRY_DATE=df.ENTRY_DATE.apply(dconvert)
    df.dob=df.dob.apply(dconvert)
    df["age"]=(df.ENTRY_DATE-df.dob).apply(dextract)/365
    #Create dictionary mapping from GRID to tuple of smallest and largest entry date in results 
    date_sets = dict()
    for i in range(len(icd)):
        d = datetime.datetime.strptime(icd.iloc[i].entry_date, "%Y-%m-%d %H:%M:%S")
        if icd.iloc[i].GRID in date_sets:
            if d < date_sets[icd.iloc[i].GRID][0]:
                date_sets[icd.iloc[i].GRID][0] = d
            if d > date_sets[icd.iloc[i].GRID][1]:
                 date_sets[icd.iloc[i].GRID][1] = d
        else:
            date_sets[icd.iloc[i].GRID] = [d,d]
    for i in range(len(df)):
        d = df.iloc[i].ENTRY_DATE
        if df.iloc[i].GRID in date_sets:
            if d < date_sets[df.iloc[i].GRID][0]:
                date_sets[df.iloc[i].GRID][0] = d
            if d > date_sets[df.iloc[i].GRID][1]:
                date_sets[df.iloc[i].GRID][1] = d
        else:
            date_sets[df.iloc[i].GRID] = [d,d]
    #Get years between first and last entry dates for GRID
    #Create column in df based off it
    df['total_years_at_v'] = df.GRID.apply(year_calc, args=(date_sets,))
    print(df.head())
    #Get distinct ICD codes
    unique_codes = icd.code.unique()
    #Use distinct ICD codes to get list of present phecodes across population
    phewas_codes = set(phecode_table[phecode_table.icd9.isin(unique_codes)].phewas_code)
    phewas_dict = dict()
    for i in range(len(phecode_table)):
        phewas_dict[phecode_table.iloc[i]['icd9']] = phecode_table.iloc[i]['phewas_code']
    #Make new dataframe with distinct GRIDS, column for number of result values
    res = df.groupby(['GRID','Result']).Result.size().unstack(fill_value=0)
    res.to_csv("test.csv")
    #res = res.groupby('GRID').sum()
    res = res.reset_index()
    print(res.head())
    #Add age and total years to res
    print('mergin')
    age_info = df[['GRID', 'age', 'total_years_at_v']]
    res = res.merge(age_info, on="GRID", how="left")
    print(res.head())
    #Create columns for counts of phecodes for each individual, initialized to zero
    for code in phewas_codes:
        res[code] = 0
    for i in range(len(icd)):
        if icd.iloc[i].code in phewas_dict:
            res.loc[res.GRID==icd.iloc[i].GRID, phewas_dict[icd.iloc[i].code]] += 1
    #res = res.drop_duplicates(subset="GRID")
    res.to_csv(sys.argv[5], index=False)
