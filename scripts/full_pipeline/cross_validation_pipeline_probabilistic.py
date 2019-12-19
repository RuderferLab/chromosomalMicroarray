import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
from scipy.stats import randint as sp_randint
import scipy.stats as st
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

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
    if (fp(y_true, y_pred) + tp(y_true, y_pred))>0:
        return tp(y_true, y_pred)/(fp(y_true, y_pred) + tp(y_true, y_pred))
    else:
        return 0

def sklearn_pipeline(df, target, cpu_num, search_method):
    #Define the pipeline
    print('Beginning pipeline')
    start = time.time()
    reduce_dim_pca = [PCA(50), PCA(75), PCA(100), PCA(), None]
    reduce_dim_umap = [umap.UMAP(n_components=5), umap.UMAP(n_components=10), umap.UMAP(n_components=15), None]
    #First selector is only the last column, second selector is the selector for all columns but the last column
    selectors = [FunctionTransformer(select_last, validate=True), FunctionTransformer(select_all_but_last, validate=True), FunctionTransformer(select_all_but_last_and_binarize, validate=True)]
    pipe = Pipeline(steps=[('column_selector', None), ('reduce_dim_1', None), ('reduce_dim_2', None), ('classify', None)])
    #Define param grid
    if search_method=='grid':
        param_grid = [
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[LogisticRegression(solver='lbfgs')],#, SVC(probability=True)],
                    'classify__C':[0.1, 1,10,100]
                },
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600]
                },
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca, 
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[GradientBoostingClassifier()],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600],
                    'classify__subsample': [0.8, 1.0]
                },
                {
                    'column_selector': [selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[BernoulliNB()],
                    'classify__alpha': [0, 1.0]
                },
                {
                    'column_selector': [selectors[1]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[GaussianNB()]
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[GaussianNB()]
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[LogisticRegression(solver='lbfgs')],#, SVC(probability=True)],
                    'classify__C':[0.1, 1,10,100]
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[GradientBoostingClassifier()],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600],
                    'classify__subsample': [0.8, 1.0]
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600]
                }
            ]
    else:
        #random search
        param_grid = [
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[LogisticRegression(solver='lbfgs')],
                    'classify__C': st.expon(scale=100)
                },
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': sp_randint(50,500),
                    'classify__min_samples_leaf': sp_randint(1,10),
                    'classify__min_samples_split': sp_randint(2,24),
                    'classify__n_estimators': sp_randint(200,800)
                },
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[GradientBoostingClassifier()],
                    'classify__max_depth': sp_randint(50,500),
                    'classify__min_samples_leaf': sp_randint(1,10), 
                    'classify__min_samples_split': sp_randint(2,24),
                    'classify__n_estimators': sp_randint(200,800),
                    'classify__subsample': [0.3,0.5,0.8,1.0]
                },
                {
                    'column_selector': [selectors[1]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[GaussianNB()]
                },
                {
                    'column_selector': [selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[BernoulliNB()],
                    'classify__alpha': st.expon(scale=.1)
                },                
                {
                    'column_selector': [selectors[0]],
                    'classify':[LogisticRegression(solver='lbfgs')],
                    'classify__C': st.expon(scale=100)
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[GradientBoostingClassifier()],
                    'classify__max_depth': sp_randint(50,500),
                    'classify__min_samples_leaf': sp_randint(1,10),
                    'classify__min_samples_split': sp_randint(2,24),
                    'classify__n_estimators': sp_randint(200,800),
                    'classify__subsample': [0.3,0.5,0.8,1.0]
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': sp_randint(50,500),
                    'classify__min_samples_leaf': sp_randint(1,10),
                    'classify__min_samples_split': sp_randint(2,24),
                    'classify__n_estimators': sp_randint(200,800)
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[GaussianNB()]
                }       
            ]
    #Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20)
    #create grid search cv with the above param_grid
    splits = KFold(n_splits=4, shuffle=True)
    score_custom = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn), 'precision_micro': 'precision_micro', 'f1': 'f1', 'auc': 'roc_auc', 'brier_score_loss': 'brier_score_loss', 'neg_log_loss': 'neg_log_loss', 'ppv': make_scorer(ppv)}
    if search_method=='grid':
        search = GridSearchCV(pipe, cv=splits, scoring=score_custom, refit='f1', param_grid=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False)
    else:
        #number of iterations = number of settings to test
        #Total runs = n_iter*cv (4 in our case)
        search = RandomizedSearchCV(pipe, cv=splits, scoring=score_custom, refit='f1', param_distributions=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False, n_iter=500)
    #Fit it all!
    print(search)
    print(pipe)
    search.fit(X_train, y_train)
    final_results_df = pd.DataFrame(search.cv_results_)
    #final_results_df.to_csv(out, index=False)
    #Show best estimator and results
    best_est = search.best_estimator_
    print(best_est)
    print('\n')
    print(search.best_params_)
    print('\n')
    print(search.best_score_)
    print('\n')
    #Calibrate best estimator
    cal_clf = CalibratedClassifierCV(best_est, method='isotonic', cv=3)
    cal_clf.fit(X_train, y_train)
    #Fit pipeline to test set with best parameters from above search
    #pipe.set_params(**search.best_params_)
    #pipe.fit(X_train, y_train)
    print('\n')
    print('Results of best estimator chosen by CV process:\n')
    preds= cal_clf.predict(X_test)#best_est.predict(X_test)#pipe.predict(X_test)
    probs = cal_clf.predict_proba(X_test)#best_est.predict_proba(X_test)[:,1]#pipe.predict_proba(X_test)[:,1]
    test_ret_df = pd.DataFrame()
    test_ret_df['target'] = y_test
    test_ret_df['case_probs'] = probs
    print(classification_report(y_test, preds))
    total = time.time()-start
    print('Elapsed time:')
    print(total)
    return final_results_df, cal_clf, test_ret_df

def calibrate_and_train(features, target, clf):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)
    cal_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    cal_clf.fit(X_train, y_train)
    probs = cal_clf.predict_proba(X_test)
    #Returns only case probabilities and targets
    return (probs[:, 1], y_test)

if __name__=='__main__':
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    #Drop rows which have no demographic info
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    #Binarize and get columns for phecodes
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    #phedf = df.loc[:, phe_list]
    #phedf[phedf>0] = 1
    #df[phe_list] = phedf
    #Run the pipeline
    frdf, _ = sklearn_pipeline(df[phe_list+['weight_sum']], df['CC_STATUS'].astype(int))
    frdf.to_csv(out, index=False)
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
