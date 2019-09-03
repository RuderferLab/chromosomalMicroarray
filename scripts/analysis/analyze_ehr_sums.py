import sys
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot(cc, nocc, outdir):
    sns.distplot(cc['BINARY_WEIGHTED_SUM'], label='Case Control weights')
    sns.distplot(nocc['BINARY_WEIGHTED_SUM'].loc[nocc.BINARY_WEIGHTED_SUM<700], label='Non case control weights').set_title("Case control vs non case control log_weights")
    plt.legend()
    plt.savefig(outdir+'cc_nocc_weights_700gated.png')
    plt.clf()

if __name__=='__main__':
    bin_sums_cc = pd.read_csv(sys.argv[1])
    bin_sums_nocc = pd.read_csv(sys.argv[2])
    plot(bin_sums_cc,bin_sums_nocc, sys.argv[3])
