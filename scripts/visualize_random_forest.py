import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree.export import export_text


#model.pkl outdir/name_forfigs wide_df phecodes
if __name__=='__main__':
    filename = sys.argv[1]
    out = sys.argv[2]
    df = pd.read_csv(sys.argv[3])
    phecodes = pd.read_csv(sys.argv[4], dtype=str)
    #Get the feature names by getting the list of phecodes that are in the dataframe
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()
    #now that the model is loaded, we need to visualize all the trees
    index = 0
    for t in model.estimators_:
        '''
        plt.clf()
        plt.figure(figsize=(8,6), dpi=300)
        tree.plot_tree(t, filled=True, feature_names=phe_list, label='root')
        plt.savefig(out+"_"+str(index)+".png")
        plt.clf()
        '''
        print('index: '+str(index)+' plotted')
        r=export_text(t, feature_names=phe_list)
        if '613.5' in r:
            print(r)
        index += 1
    print('done')
