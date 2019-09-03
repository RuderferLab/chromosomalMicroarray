import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import forestci as fci
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#args
## cc_df cc_phecodes out
if __name__=='__main__':
    #load in dataframe of cc
    cc_df = pd.read_csv(sys.argv[1])
    #load in phecodes for cc
    cc_phecodes = pd.read_csv(sys.argv[2], dtype=str)
    ##load in phecodes for freq_vis
    #fv_phecodes = pd.read_csv(sys.argv[4], dtype=str)
    #Remove CC from freq_vis if present
    #Check shape difference after removal
    #fit random forest model with predetermined hyperparams on CC
    clf = RandomForestClassifier(max_depth=150, min_samples_leaf=1, min_samples_split=4, n_estimators=600)
    #clf = RandomForestClassifier(max_depth=150, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
    #clf = RandomForestClassifier(max_depth=150, min_samples_leaf=1, min_samples_split=8, n_estimators=150)
    ###clf = RandomForestClassifier(max_depth=100, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
    #pipeline with pca and svc
    #clf = LogisticRegression(C=10)#Pipeline(steps=[('pca', PCA(n_components=100)), ('svm', SVC(C=10, probability=True))])
    cc_df=cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    cc_phe_list = [phe for phe in list(cc_phecodes.PHECODE.unique()) if phe in cc_df]
    cc_phedf = cc_df.loc[:, cc_phe_list]
    cc_phedf[cc_phedf>0] = 1
    cc_df[cc_phe_list] = cc_phedf
    #shuffle
    cc_df = cc_df.sample(frac=1)
    #Get train and test
    train, test = train_test_split(cc_df, test_size=0.2)
    #Sort cc_df columns
    #cc_df = cc_df.reindex(sorted(cc_df.columns), axis=1)
    #actually fit
    clf.fit(train[cc_phe_list], train['CC_STATUS'].astype(int))
    #Predict probabilities on test set
    predicted_probs = clf.predict_proba(test[cc_phe_list])
    test['control_prob'] = predicted_probs[:, 0]
    test['case_prob'] = predicted_probs[:, 1]
    #print(test.loc[test['Result']!='Normal', 'case_prob'].mean())
    #print(test.loc[test['Result']=='Normal', 'case_prob'].mean())
    #Save GRIDS with predicted probabilities
    out = sys.argv[3]
    test.to_csv(out, index=False)
