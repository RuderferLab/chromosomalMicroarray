import sys
import pandas as pd
from scipy.stats import fisher_exact


'''
Format for test: [[not_present_control, not_present_case],[present_control, present_case]]
'''
def phewas_wide(code_df, code_list, target_var):
    results_df = pd.DataFrame()
    odds_list = []
    pval_list = []
    not_present_control_list = []
    not_present_case_list = []
    present_control_list = []
    present_case_list = []
    for code in code_list:
        present_case = code_df.loc[(code_df[target_var]==1)&(code_df[code]>0)].shape[0]
        present_control = code_df.loc[(code_df[target_var]==0)&(code_df[code]>0)].shape[0]
        not_present_case = code_df.loc[(code_df[target_var]==1)&(code_df[code]==0)].shape[0]
        not_present_control = code_df.loc[(code_df[target_var]==0)&(code_df[code]==0)].shape[0]
        odds, pval = fisher_exact([[not_present_control, not_present_case],[present_control, present_case]])
        odds_list.append(odds)
        pval_list.append(pval)
        not_present_control_list.append(not_present_control)
        not_present_case_list.append(not_present_case)
        present_control_list.append(present_control)
        present_case_list.append(present_case)
        #print('?')
        #print(code)
        #print(pval)
        #print(odds)
        #print('!!')
    results_df['code']=code_list
    results_df['odds_ratio']=odds_list
    results_df['pval']=pval_list
    results_df['not_present_control']=not_present_control_list
    results_df['not_present_case']=not_present_case_list
    results_df['present_case']=present_case_list
    results_df['present_control']=present_control_list
    return results_df


'''
Input:
    case_control_df: Wide dataframe containing phecodes for each individual, as well as a target_var column denoting the case control status (1=case, 0=control)
    phe_list_file: file containing the name of each column which is a code. No columns, each name is on its own line.
    target_var: target variable string (ex: CC_STATUS)
    out: filepath to write results to
'''
if __name__=="__main__":
    case_control_df = pd.read_csv(sys.argv[1])
    phe_list_file = open(sys.argv[2], 'r')
    target_var = sys.argv[3]
    out = sys.argv[4]

    phe_list = []
    for line in phe_list_file:
        phe_list.append(line.strip('\n'))

    phe_list = [x for x in phe_list if x in case_control_df.columns]

    res=phewas_wide(case_control_df[phe_list+[target_var]], phe_list, target_var)
    
    res.to_csv(out, index=False)
