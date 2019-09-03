import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def importance_by_prevalence(df, phe_list, importance_df, out):
    #count up number of people per phecode
    count_dict = dict()
    for phecode in phe_list:
        if phecode in df:
            count_dict[phecode] = df[phecode].sum()/df.GRID.unique().shape[0]
    prev_df = pd.DataFrame.from_dict(count_dict, orient='index', columns=['prevalence'])
    #save prevalence df
    prev_df.to_csv('files/cc_phecode_prevalences.csv')
    prev_df = prev_df.reset_index()
    #rename prev df index to 'phecode'
    prev_df.rename(index=str, columns={'index':'phecode'}, inplace=True)
    #merge on the phecodes
    merged = importance_df.merge(prev_df, on='phecode', how='inner')
    #graph the results
    sns.lineplot(x='prevalence', y='importance', data=merged)
    plt.savefig(out)
    plt.clf()


def importance_barplot(importance_df, out):
    sns.barplot(x=importance_df.phecode[:10], y='importance', ci='sd', order=importance_df.phecode[:10], data=importance_df)
    plt.savefig(out)
    plt.clf()

#cc_df importance phecodes out
if __name__=='__main__':
    #Read in phecode df for cc
    df = pd.read_csv(sys.argv[1])
    #read in phecodes
    phecodes = pd.read_csv(sys.argv[3], dtype=str)
    #Drop na people and binarize data if not already binarized
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    #read in importance
    importance = pd.read_csv(sys.argv[2], dtype={'phecode': str})
    out = sys.argv[4]
    #run analysis
    #importance_by_prevalence(df, phe_list, importance, out)
    importance_barplot(importance, out)
