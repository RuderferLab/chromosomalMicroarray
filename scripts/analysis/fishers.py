import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chisquare, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

def chitests(ldf, weights):
    #cases = []
    #controls = []
    results = dict()
    for code in weights.keys():
        if code in ldf:
            not_present = [len(ldf[code].loc[(ldf.cc_status==0)&(ldf[code]==0)]), len(ldf[code].loc[(ldf.cc_status==1)&(ldf[code]==0)])]
            cont = len(ldf[code].loc[(ldf.cc_status==0)])-len(ldf[code].loc[(ldf.cc_status==0)&(ldf[code]==0)])
            case = len(ldf[code].loc[(ldf.cc_status==1)])-len(ldf[code].loc[(ldf.cc_status==1)&(ldf[code]==0)]) 
            present = [cont, case]
            #cases.append(case)
            #controls.append(cont)
            #chi, p = chisquare([cont,case])
            oddr, p = fisher_exact([not_present, present])
            results[code]=(oddr, p, not_present[0], not_present[1], cont, case)
        else:
            pass
            #print("NOT IN")
            #print(code)
    #print(np.array([controls,cases]).T)
    #print(chisquare(np.array([controls,cases]).T))
    #print(results.items())
    return results


'''
Same as get_sums(), but the phecode_desc is a dict mapping from phecode to the phewas_string for it (all lower case)
'''
def get_sums_no_ca(cc_df, weights, unique_phecodes, phecode_desc):
    cc_df['weight_sum'] = 0
    for phc in weights.keys():
        #Create weight sum column, where for each phecode column, if the count is nonzero the corresponding weight is added to the sum, otherwise it is ignored.
        if phc in cc_df.columns and "congenital" not in phecode_desc[phc]:
            cc_df['weight_sum_no_ca'] += np.where(cc_df[phc]>0, weights[phc], 0)
        else:
            pass
            #print(phc)
    #print(cc_df.head())
    #print(len(cc_df['weight_sum'].unique()))
    cc_df=cc_df.merge(unique_phecodes, on='GRID', how='left').fillna(0)
    cc_df['unique_phecode_adjusted_weight']=cc_df['weight_sum_no_ca']/cc_df['UNIQUE_PHECODES']
    print(min(cc_df.UNIQUE_PHECODES.unique()))
    print(cc_df.unique_phecode_adjusted_weight.unique())
    return cc_df



def save_res_table(res_dict, out):
    df=pd.DataFrame.from_dict(res_dict, orient='index')
    df.columns=['or', 'pvalue', 'control_not_present', 'case_not_present', 'control_present', 'case_present']
    df.index.name='phecode'
    print(df.head())
    df.to_csv(out)


if __name__=="__main__":
    #get phecode_desc from phecode table
    ph = pd.read_table(sys.argv[4])
    pdesc=pd.Series(ph.phewas_string.values,index=ph.phewas_code).to_dict()

    weight_df=pd.read_csv(sys.argv[2])
    weights=pd.Series(weight_df.LOG_WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    res = chitests(pd.read_csv(sys.argv[1]), weights)
    save_res_table(res, sys.argv[3])

