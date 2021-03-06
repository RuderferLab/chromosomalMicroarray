import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
from scipy.stats import chi2_contingency as chi2_c
from sklearn.preprocessing import FunctionTransformer

#Plan: Reduce full feature set using PCA to size of 50, UMAP to 10
'''
Overall plan for non nn ml pipeline
1. encode data as binary phecodes and add any other features to be used for prediction
2. split data into train and test (reserving test for the end)
3. Create nested cross validation setup
    3a. 

##
Maybe instead use estimator helper class from blog post?
Need to have estimator type as a part of the pipeline, alongside the different parameter values for each estimator
That's still just going to run gridsearchcv seperately for each estimator though

Instead, I could make pipelines seperately for each of the classifiers

###

It looks like instead I can pass *multiple* dictionaries, one for each estimator
'''

def create_pc_embedding(df, embed_cols, num_pc):
    #PCA embedding
    pca = PCA(n_components=num_pc)
    pc_embedding = pca.fit_transform(df[embed_cols])
    return (pc_embedding, pca)

def create_umap_embedding(pc_df, num_ump, met):
    ump = umap.UMAP(metric=met, n_components=num_ump)
    ump_embedding = ump.fit_transform(pc_df)
    return ump_embedding

def prediction_pipeline(predictors, df, target, num_pc, num_ump, met):
    #Pick train test split
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25)
    #Get embeddings
    pca = PCA(n_components=num_pc)
    pca.fit(X_train)
    #Fit only on X_train, transform with same fitting on test and train
    X_train_transformed = pca.transform(X_train)
    X_test_transformed = pca.transform(X_test)
    #Same with umap
    ump = umap.UMAP(metric=met, n_components=num_ump, random_state=42)
    ump.fit(X_train_transformed)
    X_train_transformed = ump.transform(X_train_transformed)
    X_test_transformed = ump.transform(X_test_transformed)
    #Test provided predictors
    for p in predictors:
        p.fit(X_train_transformed, y_train)
        print(p)
        print(confusion_matrix(y_test, p.predict(X_test_transformed)))
        print(p.score(X_test_transformed, y_test))
    #new_df = pd.DataFrame(X_train_transformed, columns=['UMP-'+str(i+1) for i in range(num_ump)])
    #new_df['target']=y_train
    return (pca, ump)

def pairplot_all(components, target, target_name,out):
    sns.set()
    comp_names=components.columns.values
    components[target_name]=target
    components.loc[components[target_name]==1,target_name]='case'
    components.loc[components[target_name]==0,target_name]='control'
    components[target_name]=components[target_name].astype(str)
    sns.pairplot(components, hue="CC_STATUS", hue_order=['control', 'case'], vars=comp_names, height=4, markers=['o', 's'], plot_kws=dict(alpha=0.1))
    plt.savefig(out+'_pairplot_'+target_name+'.png')
    plt.clf()


def pairplot_covs(df, to_compare, focus, out):
    sns.set()
    sns.pairplot(df, x_vars=focus, y_vars=to_compare, height=4, plot_kws=dict(alpha=0.1))
    plt.savefig(out+'_pairplot_focused.png')
    plt.clf()


def select_last(x):
    return x[:,-1:]

def select_all_but_last_and_binarize(x):
    return np.where(x[:,:-1]>0, 1, 0)

def select_all_but_last(x):
    return x[:,:-1]

def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): 
     return confusion_matrix(y_true, y_pred)[1, 1]

def ppv(y_true, y_pred):
    return tp(y_true, y_pred)/(fp(y_true, y_pred) + tp(y_true, y_pred))

def sklearn_pipeline(df, target, out):
    #Define the pipeline
    print('Beginning pipeline')
    start = time.time()
    reduce_dim_pca = [PCA(50), PCA(75), PCA(100), PCA(250), PCA(500), None]
    reduce_dim_umap = [umap.UMAP(n_components=5), umap.UMAP(n_components=10), umap.UMAP(n_components=15), umap.UMAP(n_components=25), umap.UMAP(n_components=40), None]
    #First selector is only the last column, second selector is the selector for all columns but the last column
    selectors = [FunctionTransformer(select_last), FunctionTransformer(select_all_but_last), FunctionTransformer(select_all_but_last_and_binarize)]
    pipe = Pipeline(steps=[('column_selector', None), ('reduce_dim_1', None), ('reduce_dim_2', None), ('classify', None)])
    #Define param grid
    param_grid = [
            {
                'column_selector': [selectors[1], selectors[2]],
                'reduce_dim_1': reduce_dim_pca,
                'reduce_dim_2': reduce_dim_umap,
                'classify':[LogisticRegression(), LinearSVC()],
                'classify__C':[0.1, 1,10,100]
            },
            {
                'column_selector': [selectors[1], selectors[2]],
                'reduce_dim_1': reduce_dim_pca,
                'reduce_dim_2': reduce_dim_umap,
                'classify':[BernoulliNB()],
                'classify__alpha':[0.1,0.5,1.0]
            },
            {
                'column_selector': [selectors[1], selectors[2]],
                'reduce_dim_1': reduce_dim_pca,
                'reduce_dim_2': reduce_dim_umap,
                'classify':[RandomForestClassifier()],
                'classify__max_depth': [50, 100, 150],
                'classify__min_samples_leaf': [1, 5],
                'classify__min_samples_split': [2, 8],
                'classify__n_estimators': [50, 150]
            },
            {
                'column_selector': [selectors[0]],
                'classify':[LogisticRegression(), LinearSVC()],
                'classify__C':[0.1, 1,10,100]
            },
            {
                'column_selector': [selectors[0]],
                'classify':[BernoulliNB()],
                'classify__alpha':[0.1,0.5,1.0]
            },
            {
                'column_selector': [selectors[0]],
                'classify':[RandomForestClassifier()],
                'classify__max_depth': [50, 100, 150],
                'classify__min_samples_leaf': [1, 5],
                'classify__min_samples_split': [2, 8],
                'classify__n_estimators': [50, 150]
            }
        ]
    #Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20)
    #create grid search cv with the above param_grid
    splits = KFold(n_splits=4, shuffle=True)
    score_custom = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn), 'precision_micro': 'precision_micro', 'f1_micro': 'f1_micro', 'auc': 'roc_auc', 'ppv': make_scorer(ppv)}
    grid = GridSearchCV(pipe, cv=splits, scoring=score_custom, refit='f1_micro', param_grid=param_grid, n_jobs=12)
    #Fit it all!
    grid.fit(X_train, y_train)
    final_results_df = pd.DataFrame(grid.cv_results_)
    final_results_df.to_csv(out, index=False)
    #Show best estimator and results
    best_est = grid.best_estimator_
    print(best_est)
    print('\n')
    print(grid.best_params_)
    print('\n')
    print(grid.best_score_)
    print('\n')
    #Fit pipeline to test set with best parameters from above search
    pipe.set_params(**grid.best_params_)
    pipe.fit(X_train, y_train)
    print('\n')
    print('Results of best estimator chosen by CV process:\n')
    print(classification_report(y_test, pipe.predict(X_test)))
    total = time.time()-start
    print('Elapsed time:')
    print(total)

if __name__=='__main__':
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    #Drop rows which have no demographic info
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    #Binarize and get columns for phecodes
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    #Run the pipeline
    sklearn_pipeline(df[phe_list+['weight_sum']], df['CC_STATUS'].astype(int), out)
    #######
    #Run process for dimensionality reduction
    #pc_emb,_ = create_pc_embedding(df, phe_list, 50)
    #pc_emb = pd.DataFrame(pc_emb)
    #ump_emb = create_umap_embedding(pc_emb, 10, 'euclidean')
    #df_reduced = pd.DataFrame(ump_emb, columns=['UMP-'+str(i+1) for i in range(10)])
    #df['UMP-1']=df_reduced['UMP-1']
    #print(df.loc[df['UMP-1'].abs()>5, ['UNIQUE_PHECODES', 'CC_STATUS', 'RECORD_LEN', 'BIRTH_DATETIME', 'RACE', 'GENDER', 'Result', 'size', 'location']])
    #print(df.loc[df['RECORD_LEN']>10000, ['UNIQUE_PHECODES', 'CC_STATUS', 'RECORD_LEN', 'BIRTH_DATETIME', 'RACE', 'GENDER', 'Result', 'size', 'location']])
    #Add relevant covariate columns if they will be used for prediction?
    #Run prediction
    #pca, ump = prediction_pipeline([GaussianNB(), LogisticRegression()], df[phe_list], df['CC_STATUS'], 50, 10, 'euclidean')
    #pc_reduced = pca.transform(df[phe_list])
    #ump_reduced = ump.transform(pc_reduced)
    #df_reduced = pd.DataFrame(ump_reduced, columns=['UMP-'+str(i+1) for i in range(10)])
    #Plot components
    #pairplot_all(df_reduced, df['CC_STATUS'], 'CC_STATUS', out)
    #df['AGE']= pd.to_datetime(df['BIRTH_DATETIME'].str[:10], format='%Y-%m-%d')
    #df['AGE']=(datetime.datetime.now()-df['AGE']).astype('timedelta64[Y]')
    #compare_cols = ['AGE', 'RECORD_LEN', 'UNIQUE_PHECODES']
    #for col in compare_cols:
    #    df_reduced[col]=df[col]
    #pairplot_covs(df_reduced, compare_cols, 'UMP-1', out)
