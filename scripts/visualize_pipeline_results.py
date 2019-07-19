import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys


#Helper functions

def extract_est(val):
    for s in ['LogisticRegression', 'RandomForestClassifier', 'LinearSVC', 'BernoulliNB']:
        if s in val:
            return s

def extract_dim_red(val1, val2):
    if 'PCA' in str(val1) and 'UMAP' in str(val2):
        return 'PCA into UMAP'
    elif 'UMAP' in str(val2):
        return 'UMAP'
    elif 'PCA' in str(val1):
        return 'PCA'
    else:
        return 'No Reduction'


def extract_input(val):
    if 'select_all_but_last' in val:
        return 'binary_matrix'
    else:
        return 'PheRS'


#Runnables

'''
Input:
    df: the results dataframe from cross validation
    out: location to write the more human readable version of the dataframe as a csv
Output: A version of the dataframe with extracted input types and estimator types, as well as dimensionality reduction columns
        However, the full param columns should not be included due to being difficult to read
'''
def clean_dataframe(df, out):
    #Extract estimator
    df['estimator_type']=df['param_classify'].map(extract_est)
    #Extract input
    df['input_type'] = df['param_column_selector'].map(extract_input)
    #Extract dimensionality reductions
    df['dimensionality_reduction'] = df.apply(lambda x: extract_dim_red(x['param_reduce_dim_1'], x['param_reduce_dim_2']), axis=1)
    #Select only relevant columns for writing
    relevant_df = df[['dimensionality_reduction', 'input_type', 'estimator_type', 'mean_test_ppv', 'mean_test_precision_micro', 'mean_test_auc', 'mean_test_tp', 'mean_test_fp', 'mean_test_tn', 'mean_test_fn']]
    relevant_df.to_csv(out, index=False)

def estimator_visualization(df, out):
    #Get the specific estimator type and label it in a column
    df['estimator_type']=df['param_classify'].map(extract_est)
    #Plot each score based off the established column above in a violin plot
    for score in ['precision_weighted', 'f1_weighted', 'roc_auc']:
        sns.violinplot(x='estimator_type', y='mean_test_'+score, data=df)
        plt.savefig(out+'_'+score+'.png')
        plt.clf()

def est_specif_dimen_visualization(df, estimator_name, out):
    #Get the specific estimator type and label in column
    df['estimator_type'] = df['param_classify'].map(extract_est)
    #Get specific combination of dimensionality reductions applied and enter it in a column
    df['Dimensionality Reduction'] = df.apply(lambda x: extract_dim_red(x['param_reduce_dim_1'], x['param_reduce_dim_2']), axis=1)
    #Set scores to explore over
    scores = ['f1']#['precision_weighted', 'f1_weighted', 'roc_auc']
    for score in scores:
        ax=sns.boxplot(x='Dimensionality Reduction', y='mean_test_'+score, data=df.loc[df.estimator_type==estimator_name])
        ax.set(ylabel='Mean F1 Score')
        plt.savefig(out+'_'+score+'.png')
        plt.clf()


def input_select_vis(df, out):
    #Get the input type and save it in a column
    df['in_type'] = df['param_column_selector'].map(extract_input)
    scores = ['precision_micro', 'auc', 'ppv']
    for score in scores:
        sns.violinplot(x='in_type', y='mean_test_'+score, data=df)
        plt.savefig(out+'_'+score+'.png')
        plt.clf()

def est_input_vis(df, out):
    #Get the input type and save it in a column
    df['in_type'] = df['param_column_selector'].map(extract_input)
    #Get the estimator type and save in a column
    df['estimator_type'] = df['param_classify'].map(extract_est)
    scores = ['precision_micro', 'auc', 'ppv']
    for score in scores:
        sns.boxplot(x='estimator_type', y='mean_test_'+score, hue='in_type', data=df)
        plt.savefig(out+'_'+score+'.png')
        plt.clf()

def case_prob_by_phecode(df, phecodes, focus_code, label_code,  out):
    #Locate individuals of interest
    focused = set(phecodes.loc[phecodes.CODE==focus_code, 'GRID'])
    #Label individuals with the phecode of interest
    df[label_code] = 0
    df.loc[df.GRID.isin(focused), label_code] = 1
    sns.distplot(df.loc[df[label_code]==0], label="No Code Present")
    sns.distplot(df.loc[df[label_code]==1], label="Code Present")
    plt.savefig(out+'.png')
    plt.clf()


if __name__=='__main__':
    print('starting')
    #Read in results of cv as dataframe
    df = pd.read_csv(sys.argv[1])
    #phecodes
    #phecodes = pd.read_csv(sys.argv[2], dtype=str)
    #Out variable should be the prefix for the following plots to be generated
    out = sys.argv[2]
    ##Run estimator visualization
    #estimator_visualization(df, out)
    ##Run dimension_reduction visualization
    est_specif_dimen_visualization(df, 'RandomForestClassifier', out)
    #input_select_vis(df, out)
    #est_input_vis(df, out)
    #clean_dataframe(df, out)
    #case_prob_by_phecode(df, phecodes, '613.5', '613.5', out)
