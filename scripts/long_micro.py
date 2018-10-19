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

def uc_mark(x):
    if "unclear" in x.lower():
        return 1
    else:
        return 0

if __name__=="__main__":
    df = read_csv(sys.argv[1], quotechar="\"", escapechar="\\")
    print(df.head())
    #Filter out all duplicates
    df = df.drop_duplicates(subset="GRID")
    #Mark all unclear results
    df['unclear'] = df['result_text'].apply(uc_mark)
    #Load in ICD for given GRIDS
    icd = read_csv(sys.argv[2])
    #Load in phecode translation
    phecode_table = read_table(sys.argv[3])
    #Load dobs
    dobs = read_csv(sys.argv[4])
    #merge dobs
    df = df.merge(dobs, on="GRID", how="inner")
    #Get ages
    df.ENTRY_DATE = df.ENTRY_DATE.apply(dconvert)
    df.dob = df.dob.apply(dconvert)
    df["age"]=(df.ENTRY_DATE-df.dob).apply(dextract)/365
    #Load total years
    total_years = read_csv(sys.argv[5])
    total_years['years'] = total_years['diff']/365
    total = total_years[['GRID','years']]
    #load all events
    #all_events = read_csv(sys.argv[5])
    #Get total years using all_events
    #date_sets = dict()
    #preset date_sets using test entry date
    #for i in range(len(df)):
    #    date_sets[df.iloc[i].GRID] = [df.iloc[i].ENTRY_DATE, df.iloc[i].ENTRY_DATE]
    #for i in range(len(all_events)):
    #    if all_events.iloc[i].DATE != "None":
    #        d = datetime.datetime.strptime(all_events.iloc[i].DATE, "%Y-%m-%d")
    #        if all_events.iloc[i].GRID in date_sets:
    #            if d < date_sets[all_events.iloc[i].GRID][0]:
    #                date_sets[all_events.iloc[i].GRID][0] = d
    #            if d > date_sets[all_events.iloc[i].GRID][1]:
    #                date_sets[all_events.iloc[i].GRID][1] = d
    #        else:
    #            pass
                #date_sets[all_events.iloc[i].GRID] = [d,d]
    #Set total years
    df = df.merge(total, on="GRID", how="inner")
    #Get phecode translations
    unique_codes = icd.code.unique()
    phewas_codes = set(phecode_table[phecode_table.icd9.isin(unique_codes)].phewas_code)
    phewas_dict = dict()
    for i in range(len(phecode_table)):
        phewas_dict[phecode_table.iloc[i]['icd9']] = phecode_table.iloc[i]['phewas_code']
    for code in phewas_codes:
        df[code] = 0
    for i in range(len(icd)):
        if icd.iloc[i].code in phewas_dict:
            df.loc[df.GRID==icd.iloc[i].GRID, phewas_dict[icd.iloc[i].code]] += 1
    print(df)
    df.to_csv(sys.argv[6], index=False)
