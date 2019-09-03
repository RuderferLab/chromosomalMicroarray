import numpy as np
import pandas as  pd
import sys
import pickle
from treeinterpreter import treeinterpreter as ti

#df phecodes clf_path grid
if __name__=='__main__':
    df = pd.read_csv(sys.argv[1])
    phecode = pd.read_csv(sys.argv[2], dtype=str)
    clf = pickle.load(open(sys.argv[3], 'rb'))
    grid = sys.argv[4]
    #goal: visualize decision path across all estimators for one specific sample
    phe_list = list(phecode.PHECODE.unique())
    #phecodes = df[phe_list].copy(deep=True)
    #rf contributions by feature for grid
    print(clf.predict_proba(df.loc[df.GRID==grid, phe_list]))
    prediction, bias, contributions = ti.predict(clf, df.loc[df.GRID==grid, phe_list])
    print("Prediction: "+str(prediction))
    print("Bias: " +str(bias))
    print("Sorted contributions:")
    #for c, feature in zip(contributions[0], phe_list):
    #    print(str(feature)+' '+str(c))
    zipped = sorted(zip(contributions[0], phe_list), key=lambda x: max(x[0]))
    for c, feature in zipped:
        print(str(feature)+' '+str(c))
