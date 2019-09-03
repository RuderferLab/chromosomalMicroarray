import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

#args
## cc_df cc_phecodes out
if __name__=='__main__':
    #load in dataframe of cc
    cc_df = pd.read_csv(sys.argv[1])
    #load in phecodes for cc
    cc_phecodes = pd.read_csv(sys.argv[2], dtype=str)
    #load in phecodes for freq_vis
    #fit random forest model with predetermined hyperparams on CC
    clf = RandomForestClassifier(max_depth=150, min_samples_leaf=1, min_samples_split=8, n_estimators=150)
    ###clf = RandomForestClassifier(max_depth=100, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
    cc_df=cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    cc_phe_list = [phe for phe in list(cc_phecodes.PHECODE.unique()) if phe in cc_df]
    cc_phedf = cc_df.loc[:, cc_phe_list]
    cc_phedf[cc_phedf>0] = 1
    cc_df[cc_phe_list] = cc_phedf
    #shuffle
    cc_df = cc_df.sample(frac=1) 
    #Sort cc_df columns
    cc_df = cc_df.reindex(sorted(cc_df.columns), axis=1)
    #actually fit
    clf.fit(cc_df[cc_phe_list], cc_df['CC_STATUS'].astype(int))
    #Get importances
    importances = clf.feature_importances_
    important_features = pd.Series(data=importances,index=cc_df[cc_phe_list].columns)
    important_features.sort_values(ascending=False,inplace=True)
    #Save labelled importances
    out = sys.argv[3]
    important_features.to_csv(out)
    #Plot importances with error bars
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
            axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(cc_phe_list))[:10], 
            importances[indices][:10],
            color="r", 
            yerr=std[indices][:10],
            align="center")
    plt.xticks(range(len(cc_phe_list))[:10], indices[:10])
    #plt.xlim([-1, len(cc_phe_list)])
    plt.savefig("figs/rf_bar_importance_stderr.png")
