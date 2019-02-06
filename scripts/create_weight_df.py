import sys
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
input: 
    cc_df: Long form df of phecode counts for case control dataset
    weights: dict of form phecode: weights
'''
def get_sums(cc_df, weights, unique_phecodes):
    cc_df['weight_sum'] = 0
    for phc in weights.keys():
        #Create weight sum column, where for each phecode column, if the count is nonzero the corresponding weight is added to the sum, otherwise it is ignored.
        if phc in cc_df.columns:
            cc_df['weight_sum'] += np.where(cc_df[phc]>0, weights[phc], 0)
        else:
            pass
            #print(phc)
    #print(cc_df.head())
    #print(len(cc_df['weight_sum'].unique()))
    cc_df=cc_df.merge(unique_phecodes, on='GRID', how='left').fillna(0)
    cc_df['unique_phecode_adjusted_weight']=cc_df['weight_sum']/cc_df['UNIQUE_PHECODES']
    return cc_df

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


def has_child(code, present_codes, ancestor_dict):
    anc = [x for x in ancestor_dict[code] where x != code]
    present_ancestors = set(anc).intersection(present_codes)
    if len(present_ancestors)==0:
        return False
    else:
        return True

def remove_ancestors(s, ancestor_dict):
    #Create set of all phecode present in this individual
    index_names = s.index()
    nonzero_indices = s.nonzero()[0]
    nonzero_cols=index_names[nonzero_indices]
    #Create set of phecodes to set to zero -- include in this list if they have a child
    for col in nonzero_cols:
        if has_child(col, nonzero_cols, ancestor_dict):
            s[col]=0
    return s

'''
input:
    mappings: dictionary with an entry of each unique code, leading to all its children
    df: full long form dataframe
output:
    modified form of the df where only the most specific codes are kept -- for example, if the code structure from a unique code is shown as a tree, only the leaves remain
strategy: For every given subject run the following algorithm:
    for each code, if a child code of that code (other than itself) is present, remove that code
    This should leave us with only the leaves
'''
def leaf_select_codes(mappings, df, ancestor_dict, phecode_list):
    df[phecode_list] = df[phecode_list].apply(remove_ancestors, args=(ancestor_dict))

#long_df weight_df unique_phecodes summary_df genetic_diseases sd_clinic_bin
if __name__=="__main__":
    #Generate weight dataframe
    long_df = pd.read_csv(sys.argv[1])
    weight_df = pd.read_csv(sys.argv[2])
    weights = pd.Series(weight_df.LN.values,index=weight_df.PHECODE.astype(str)).to_dict()
    unique_phecodes = pd.read_csv(sys.argv[3])
    print('loaded')

    summed_df = get_sums(long_df, weights, unique_phecodes)
    print('got sums')
    ##Read in covariates from summary df
    ##summary_df = pd.read_csv(sys.argv[4])
    #Load in gold standard genetic diseases and genetics clinic info, merge them on by GRID
    genetic_diseases = pd.read_csv(sys.argv[4])
    clinic_bin = pd.read_table(sys.argv[5])
    #long_df weight_df unique_phecodes summary genetic_diseases SD_clinic
    #Merge them all on GRID
    for df in [genetic_diseases, clinic_bin]:#[summary_df, genetic_diseases, clinic_bin]:
        summed_df = summed_df.merge(df, how="left")
    print(summed_df.head())
    summed_df.to_csv("all_merged.csv", index=False)
