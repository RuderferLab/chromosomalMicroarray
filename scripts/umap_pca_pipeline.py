import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report
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

def sklearn_pipeline(df, target, out):
    #Define the pipeline
    print('Beginning pipeline')
    start = time.time()
    pca = PCA()
    ump = umap.UMAP()
    pipe = Pipeline(steps=[('pca', pca), ('umap', ump), ('classify', None)])
    #Define param grid
    UMAP_OPTIONS = [5, 10, 15]
    PCA_OPTIONS = [50, 80, 100]
    param_grid = [
            {
                'classify':[LogisticRegression(), LinearSVC()],
                'pca__n_components':PCA_OPTIONS,
                'umap__n_components':UMAP_OPTIONS,
                'classify__C':[0.1,1,10,100]
            },
            {
                'classify':[BernoulliNB()],
                'pca__n_components':PCA_OPTIONS,
                'umap__n_components':UMAP_OPTIONS,
                'classify__alpha':[0,0.5,1.0]
            },
            {
                'classify':[RandomForestClassifier()],
                'pca__n_components':PCA_OPTIONS,
                'umap__n_components':UMAP_OPTIONS,
                'classify__max_depth': [50, 70, 100, 150],
                'min_samples_leaf': [1, 3, 5],
                'min_samples_split': [2, 4, 8],
                'n_estimators': [50, 100, 200, 300]
            }
        ]
    #Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20)
    #create grid search cv with the above param_grid
    splits = KFold(n_splits=8, shuffle=True)
    grid = GridSearchCV(pipe, cv=splits, scoring='f1')
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
    pipe.set_params(grid.best_params_)
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
    sklearn_pipeline(df[phe_list], df['CC_STATUS'], out)
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
