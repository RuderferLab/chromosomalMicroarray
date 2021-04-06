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


def create_pc_embedding(df, embed_cols, num_pc):
    #PCA embedding
    pca = PCA(n_components=num_pc)
    pc_embedding = pca.fit_transform(df[embed_cols])
    return (pc_embedding, pca)

def create_umap_embedding(pc_df, num_ump, met):
    ump = umap.UMAP(metric=met, n_components=num_ump)
    ump_embedding = ump.fit_transform(pc_df)
    return ump_embedding


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
    reduce_dim_pca = [PCA(.95), None]
    reduce_dim_umap = [umap.UMAP(n_components=10), None]
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
                    'classify__learning_rate': [0.01,0.1,0.2],
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
                    'classify__learning_rate': [0.01,0.1,0.2],
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
    elif search_method=='small_grid':
        param_grid = [
                {
                    'column_selector': [selectors[1]],
                    'classify':[LogisticRegression(solver='lbfgs')],
                    'classify__C':[0.1, 1,10,100]
                },
                {
                    'column_selector': [selectors[1]],
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600]
                },
                {
                    'column_selector': [selectors[1]],
                    'classify':[GradientBoostingClassifier()],
                    'classify__learning_rate': [0.01,0.1,0.2],
                    'classify__max_depth': [100, 150, 200],
                    'classify__min_samples_leaf': [1, 5],
                    'classify__min_samples_split': [2, 8],
                    'classify__n_estimators': [600],
                    'classify__subsample': [0.8, 1.0]
                },
                {
                    'column_selector': [selectors[1]],
                    'classify':[GaussianNB()]
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
                    'classify__max_depth': sp_randint(3,250),
                    'classify__min_samples_leaf': sp_randint(1,200),
                    'classify__max_leaf_nodes':sp_randint(10,200),
                    'classify__min_samples_split': sp_randint(2,200),
                    'classify__n_estimators': sp_randint(200,1100)
                },
                {
                    'column_selector': [selectors[1], selectors[2]],
                    'reduce_dim_1': reduce_dim_pca,
                    'reduce_dim_2': reduce_dim_umap,
                    'classify':[GradientBoostingClassifier()],
                    'classify__learning_rate': st.reciprocal(1e-3, 5e-1),
                    'classify__max_depth': sp_randint(3,250),
                    'classify__max_leaf_nodes':sp_randint(10,200),
                    'classify__min_samples_leaf': sp_randint(1,200), 
                    'classify__min_samples_split': sp_randint(2,200),
                    'classify__n_estimators': sp_randint(200,1100),
                    'classify__subsample': st.uniform(0.5,0.5)
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
                    'classify__learning_rate': st.reciprocal(1e-3, 5e-1),
                    'classify__max_depth': sp_randint(3,250),
                    'classify__max_leaf_nodes':sp_randint(10,200),
                    'classify__min_samples_leaf': sp_randint(1,200), 
                    'classify__min_samples_split': sp_randint(2,200),
                    'classify__n_estimators': sp_randint(200,1100),
                    'classify__subsample': st.uniform(0.5,0.5)
                },
                {
                    'column_selector': [selectors[0]],
                    'classify':[RandomForestClassifier()],
                    'classify__max_depth': sp_randint(3,250),
                    'classify__max_leaf_nodes':sp_randint(10,200),
                    'classify__min_samples_leaf': sp_randint(1,200),
                    'classify__min_samples_split': sp_randint(2,200),
                    'classify__n_estimators': sp_randint(200,1100)
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
    score_custom = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn), 'precision_micro': 'precision_micro', 'f1': 'f1', 'auc': 'roc_auc', 'brier_score_loss': 'brier_score_loss', 'neg_log_loss': 'neg_log_loss', 'ppv': make_scorer(ppv), 'average_precision': 'average_precision'}
    if search_method=='grid':
        search = GridSearchCV(pipe, cv=splits, scoring=score_custom, refit='average_precision', param_grid=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False)
    elif search_method=='small_grid':
        search = GridSearchCV(pipe, cv=splits,
                scoring={'precision_micro': 'precision_micro', 'f1_micro': 'f1_micro'},
                refit='f1_micro', param_grid=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False)
    else:
        #number of iterations = number of settings to test
        #Total runs = n_iter*cv (4 in our case)
        search = RandomizedSearchCV(pipe, cv=splits, scoring=score_custom, refit='f1', param_distributions=param_grid, n_jobs=cpu_num, pre_dispatch=2*cpu_num, return_train_score=False, n_iter=5000)
    #Fit it all!
    print(search)
    print(pipe)
    #print(X_train.drop('GRID',axis=1))
    #print(select_all_but_last(X_train).drop('GRID',axis=1))
    search.fit(X_train.drop('GRID',axis=1), y_train)
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
    cal_clf.fit(X_train.drop('GRID',axis=1), y_train)
    #Fit pipeline to test set with best parameters from above search
    #pipe.set_params(**search.best_params_)
    #pipe.fit(X_train, y_train)
    print('\n')
    print('Results of best estimator chosen by CV process:\n')
    preds = cal_clf.predict(X_test.drop('GRID',axis=1))
    probs = cal_clf.predict_proba(X_test.drop('GRID',axis=1))
    if search_method != 'small_grid':
        probs = probs[:,1]
    test_ret_df = pd.DataFrame()
    test_ret_df['target'] = y_test
    if search_method == 'small_grid':
        index = 0
        for cls in cal_clf.classes_:
            test_ret_df[str(cls)+'_prob']=probs[:,index]
            index+=1
    else:
        test_ret_df['case_probs'] = probs
    test_ret_df['preds'] = preds
    test_ret_df['GRID']=X_test['GRID']
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

def run_pipeline_matched_df():
    #load in prematched cc df, weights file (for accessing in specified order), output, key for searching, and target
    wide_df = pd.read_csv(sys.argv[1])
    weight_df = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    key = sys.argv[4]
    target = sys.argv[5]
    cpu_num = int(sys.argv[6])
    cog_anom = list(pd.read_csv(sys.argv[7], dtype=str).PHECODE)
    #add dummy weight sum
    wide_df['dummy_weight_sum']=0.0
    #remove cheat codes
    phe_list = [x for x in list(weight_df.PHECODE.unique()) if x in wide_df.columns and x not in cog_anom]# ['758','758.1','759','759.1']]
    frdf, cal_clf, test_ret_df = sklearn_pipeline(wide_df[['GRID']+phe_list+['dummy_weight_sum']], wide_df[target], cpu_num, key)
    frdf.to_csv(out+'_'+target+'_final_results_df_no_cog_anom.csv',index=False)
    test_ret_df.to_csv(out+'_'+target+'_test_set_df_no_cog_anom.csv',index=False)
    from joblib import dump
    dump(cal_clf, out+'_'+target+'_classfier_no_cog_anom.joblib')

if __name__=='__main__':
    run_pipeline_matched_df()
