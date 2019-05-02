import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#args
## cc_df freq_vis_df cc_phecodes freq_vis_phecodes out clf_out
if __name__=='__main__':
    #load in dataframe of cc
    cc_df = pd.read_csv(sys.argv[1])
    #load in dataframe of freq_vis
    fv_df = pd.read_csv(sys.argv[2])
    #load in phecodes for cc
    cc_phecodes = pd.read_csv(sys.argv[3], dtype=str)
    #load in phecodes for freq_vis
    fv_phecodes = pd.read_csv(sys.argv[4], dtype=str)
    #Remove CC from freq_vis if present
    print(fv_df.shape)
    fv_df = fv_df.loc[~fv_df.GRID.isin(cc_df.GRID)]
    #Check shape difference after removal
    print(fv_df.shape)
    #fit random forest model with predetermined hyperparams on CC
    clf = RandomForestClassifier(max_depth=150, min_samples_leaf=1, min_samples_split=8, n_estimators=150)
    cc_df=cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    cc_phe_list = [phe for phe in list(cc_phecodes.PHECODE.unique()) if phe in cc_df]
    cc_phedf = cc_df.loc[:, cc_phe_list]
    cc_phedf[cc_phedf>0] = 1
    cc_df[cc_phe_list] = cc_phedf
    #shuffle
    cc_df = cc_df.sample(frac=1) 
    clf.fit(cc_df[cc_phe_list], cc_df['CC_STATUS'].astype(int))
    #Get freq_vis in proper format (binary matrix), with only phecodes which are in both dfs (can't predict on phecodes we haven't seen before in the training set)
    fv_df = fv_df.drop(fv_df[fv_df.BIRTH_DATETIME=='0'].index)
    fv_phe_list = [phe for phe in list(fv_phecodes.PHECODE.unique()) if phe in fv_df and phe in cc_df]
    fv_phedf = fv_df.loc[:, fv_phe_list]
    fv_phedf[fv_phedf>0] = 1
    fv_df[fv_phe_list] = fv_phedf
    #shuffle
    fv_df = fv_df.sample(frac=1)
    #Predict probabilities on freq_vis
    predicted_probs = clf.predict_proba(fv_df[fv_phe_list])
    fv_df['control_prob'] = predicted_probs[:, 0]
    fv_df['case_prob'] = predicted_probs[:, 1]
    #Save GRIDS with predicted probabilities
    out = sys.argv[5]
    fv_df[:, ['GRID', 'control_prob', 'case_prob']].to_csv(out)
    #Save pickle of classifier
    clf_out = sys.argv[6]
    handle = open(clf_out, 'wb')
    pickle.dump(clf, handle)
    handle.close()
