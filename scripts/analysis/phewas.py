import sys
import pandas as pd
from scipy.stats import fisher_exact


'''
Format for test: [[not_present_control, not_present_case],[present_control, present_case]]
'''
def phewas_wide(code_df, code_list):
    results_df = pd.DataFrame()
    odds_list = []
    pval_list = []
    for code in code_list:
        present_case = code_df.loc[(code_df['CC_STATUS']==1)&(code_df[code]>0)].shape[0]
        present_control = code_df.loc[(code_df['CC_STATUS']==0)&(code_df[code]>0)].shape[0]
        not_present_case = code_df.loc[(code_df['CC_STATUS']==1)&(code_df[code]==0)].shape[0]
        not_present_control = code_df.loc[(code_df['CC_STATUS']==0)&(code_df[code]==0)].shape[0]
        odds, pval = fisher_exact([[not_present_control, not_present_case],[present_control, present_case]])
        odds_list.append(odds)
        pval_list.append(pval)
    results_df['code']=code_list
    results_df['odds_ratio']=odds_list
    results_df['pval']=pval_list
    return results_df


'''
Input:
    case_control_df: Wide dataframe containing phecodes for each individual, as well as a CC_STATUS column denoting the case control status (1=case, 0=control)
    phe_list_file: file containing the name of each column which is a code. No columns, each name is on its own line.
    out: filepath to write results to
'''
if __name__=="__main__":
    case_control_df = pd.read_csv(sys.argv[1])
    phe_list_file = open(sys.argv[2], 'r')
    out = sys.argv[3]

    phe_list = []
    for line in phe_list_file:
        phe_list.append(line.strip('\n'))

    res=phewas_wide(case_control_df[phe_list+'CC_STATUS'], phe_list)
    
    res.to_csv(out)
