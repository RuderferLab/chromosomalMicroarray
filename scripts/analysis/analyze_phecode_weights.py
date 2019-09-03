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
    print(min(cc_df.UNIQUE_PHECODES.unique())) 
    print(cc_df.unique_phecode_adjusted_weight.unique())
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

def plot_sums(sum_df, outdir):
    #sns.distplot(sum_df['weight_sum']).set_title("All case control subjects")
    sns.distplot(sum_df['weight_sum'].loc[sum_df['cc_status']==0], label='Controls', color="r")
    sns.distplot(sum_df['weight_sum'].loc[sum_df['cc_status']==1], label='Cases', color="teal").set_title("Both case and control")
    plt.legend()
    plt.savefig(outdir+'both_weights.png')
    plt.clf()
    sns.distplot(sum_df['weight_sum'].loc[sum_df['cc_status']==0]).set_title("Controls")
    print(sum_df['weight_sum'].loc[sum_df['cc_status']==0].mean())
    plt.savefig(outdir+'control_weights.png')
    plt.clf()
    sns.distplot(sum_df['weight_sum'].loc[sum_df['cc_status']==1]).set_title("Cases")
    print(sum_df['weight_sum'].loc[sum_df['cc_status']==1].mean())
    plt.savefig(outdir+'case_weights.png')
    plt.clf()
    #unique_phecodes weighting
    sum_df=sum_df.fillna(0)
    sns.distplot(sum_df['unique_phecode_adjusted_weight'].loc[sum_df['cc_status']==0], label='Controls', color="r")
    sns.distplot(sum_df['unique_phecode_adjusted_weight'].loc[sum_df['cc_status']==1], label='Cases', color="teal").set_title("Unique phecode adjustment: Both case and control")
    plt.legend()
    plt.savefig(outdir+'both_weights_adjusted.png')
    plt.clf()
    #Abnormal normal difference
    #print(sum_df['weight_sum'].loc[(sum_df['cc_status']==1) & (sum_df['Result']=="Normal")])
    sns.distplot(sum_df['weight_sum'].loc[(sum_df['cc_status']==1)&(sum_df['Result']=="Normal")], label='Normal Cases')
    sns.distplot(sum_df['weight_sum'].loc[(sum_df['cc_status']==1)&(sum_df['Result']!="Normal")], label='Abnormal Cases')
    plt.legend()
    plt.savefig(outdir+"abnormal_normal_case_sum_dist.png")
    plt.clf()

if __name__=="__main__":
    long_df = pd.read_csv(sys.argv[1])
    weight_df = pd.read_csv(sys.argv[2])
    weights = pd.Series(weight_df.LOG_WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    unique_phecodes = pd.read_csv(sys.argv[3])

    summed_df = get_sums(long_df, weights, unique_phecodes)

    plot_sums(summed_df, sys.argv[4])
