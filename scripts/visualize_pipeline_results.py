import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def extract_est(val):
    for s in ['LogisticRegression', 'RandomForestClassifier', 'LinearSVC', 'BernoulliNB']:
        if s in val:
            return s

def extract_dim_red(val1, val2):
    dim_1 = None
    dim_2 = None
    if str(val1)=='nan':
        dim_1 = None
    elif 'PCA' in val1:
        dim_1 = 'PCA'

    if str(val2)=='nan':
        dim_2 = None
    elif 'UMAP' in val2:
        dim_2 = 'UMAP'
    return str(dim_1)+'_'+str(dim_2)

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
    df['dimensionality_reduction'] = df.apply(lambda x: extract_dim_red(x['param_reduce_dim_1'], x['param_reduce_dim_2']), axis=1)
    #Set scores to explore over
    scores = ['precision_weighted', 'f1_weighted', 'roc_auc']
    for score in scores:
        sns.violinplot(x='dimensionality_reduction', y='mean_test_'+score, data=df.loc[df.estimator_type==estimator_name])
        plt.savefig(out+'_'+score+'.png')
        plt.clf()

if __name__=='__main__':
    #Read in results of cv as dataframe
    df = pd.read_csv(sys.argv[1])
    #Out variable should be the prefix for the following plots to be generated
    out = sys.argv[2]
    ##Run estimator visualization
    #estimator_visualization(df, out)
    ##Run dimension_reduction visualization
    est_specif_dimen_visualization(df, 'RandomForestClassifier', out)
