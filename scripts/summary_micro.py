import pandas as pd
import sys
import datetime

#NOTE_ID,GRID,ENTRY_DATE,Result,size,location,result_text,interp,indication,Method,BP_start,BP_end,ISCN,unclear,dob,age,years
def dconvert(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S")

def create_summary():
    df = pd.read_csv(sys.argv[1])
    #File containing GRID, PHECODE, DATE format for each phecode present for each GRID
    phecodes = pd.read_csv(sys.argv[2])
    records = pd.read_csv(sys.argv[3])
    records['record_len'] = records['record_len'].fillna(0).astype(int)
    #Create summary dataframe
    summary = pd.DataFrame()
    summary["GRID"] = df.GRID
    summary['age'] = df.age
    summary['dob'] = df.dob
    summary['result'] = df.Result
    summary['location'] = df.location
    summary['size'] = df['size']
    summary['unclear'] = df.unclear
    summary["phecode_sum"] = 0#df.drop(['NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','unclear','dob','age','years'],axis=1).sum(axis=1)
    summary['pretest_sum'] = 0
    phecodes['date'] = phecodes.date.apply(dconvert)#datetime.datetime.strptime(phecodes.date, "%Y-%m-%d %H:%M:%S")
    for grid in summary.GRID.values:
        test_date = datetime.datetime.strptime(df.loc[df.GRID==grid].ENTRY_DATE.values[0], "%Y-%m-%d")
        g_sum = len(phecodes.loc[(phecodes.GRID == grid) & (phecodes.date<test_date)])
        summary.loc[summary.GRID==grid, 'pretest_sum']=g_sum
        summary.loc[summary.GRID==grid, 'phecode_sum']=len(phecodes.loc[phecodes.GRID == grid])
        summary.loc[summary.GRID==grid, 'pretest_phecodes_unique']=len(phecodes.loc[(phecodes.GRID == grid) & (phecodes.date<test_date)].code.unique())
        summary.loc[summary.GRID==grid, 'phecodes_unique']=len(phecodes.loc[phecodes.GRID == grid].code.unique())
        summary.loc[summary.GRID==grid, 'unique_categories'] = len(phecodes.loc[phecodes.GRID == grid].cat.unique())
    summary['pretest_sum'] = summary['pretest_sum'].astype(int)
    summary['phecode_sum'] = summary['phecode_sum'].astype(int)
    summary['pretest_phecodes_unique'] = summary['pretest_phecodes_unique'].astype(int)
    summary['phecodes_unique'] = summary['phecodes_unique'].astype(int)
    summary['unique_categories'] = summary['unique_categories'].astype(int)
    summary = summary.merge(records, how='left', on='GRID')
    #summary.to_csv(sys.argv[3], index=False)
    #summary['interp'] = df.interp
    summary['indication'] = df.indication
    summary.to_csv(sys.argv[4], index=False)


if __name__=="__main__":
    create_summary()
