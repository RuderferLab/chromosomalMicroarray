import pandas as pd
import datetime
import sys


def dconvert(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d")

def ddays(d):
    return int(round(d.days/365))

'''
Function: Given input of "cases" in the form of microarray subjects.
        Need to match the cases with a set of controls from the full population, who are not already in the cases, and who are not already selected.
        Should be matched 4:1 on age, sex, record_length, and number of unique years in treatment record.
input: full_pop and cases, both dataframes, which contain grid, gender, age, record_len(in days), and unique_years
'''
def cc_match(full_pop, cases):
    #full_pop['cc_status']=0
    #full_pop.loc[full_pop.GRID.isin(cases.GRID), 'cc_status']=1
    #drop all cases from fullpop
    all_control = full_pop.loc[~full_pop.GRID.isin(cases.GRID)]
    new_ds_idx=[]
    #select 4 controls for each case, by a step process
    #Remove selected controls from all_control, and add them to a list of new case indices. After selection is complete, copy indices in new_cases from full_pop
    for i in range(len(cases)):
        matching = all_control.loc[(all_control.GENDER==cases.iloc[i].GENDER) & (all_control.age==cases.iloc[i].age) & (all_control.UNIQUE_YEARS==cases.iloc[i].UNIQUE_YEARS)]
        matching['ab'] = abs(matching['RECORD_LENGTH_DAYS']-cases.iloc[i].RECORD_LENGTH_DAYS)
        matching=matching.sort_values(by='ab')
        final = matching[:4]
        new_ds_idx.extend(final.index)
        all_control=all_control.drop(final.index)
    new_ds_idx.extend(full_pop.loc[full_pop['GRID'].isin(cases['GRID'])].index)
    case_control = full_pop.iloc[new_ds_idx,:]
    return case_control
    #case_control.to_csv(out, index=False)

#fullpop cases out
if __name__=='__main__':
    fp=pd.read_csv(sys.argv[1])
    cases = pd.read_csv(sys.argv[2])
    
    fp['age'] = datetime.datetime.now()-fp.BIRTH_DATETIME.str[:10].apply(dconvert)
    fp['age'] = fp['age'].apply(ddays)
    cases['age'] = datetime.datetime.now()-cases.BIRTH_DATETIME.str[:10].apply(dconvert)
    cases['age'] = cases['age'].apply(ddays)
    fp['cc_status']=0
    fp.loc[fp.GRID.isin(cases.GRID), 'cc_status']=1

    #print(cases.age)

    res=cc_match(fp, cases)#, sys.argv[3])
    res.to_csv(out, index=False)
