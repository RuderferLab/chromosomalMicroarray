print('reached start of script')
import numpy as np
from random import shuffle
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
import statistics
import pickle


def biased_sample():
    if np.random.random()<=0.8:
        return np.random.uniform(0,0.5)
    else:
        return np.random.uniform(0.5,1.0)


#Return the roc_auc_score for a given run of assigning probabilities
def biased_test(df):
    nums = []
    for i in range(df.GRID.shape[0]):
        nums.append(biased_sample())
    df['biased_random']=nums
    return roc_auc_score(df['gClin'], df['biased_random'])

def uniform_prb_test(df):
    nums = np.random.random(df.shape[0])
    df['uniform_random']=nums
    return roc_auc_score(df['gClin'], df['uniform_random'])


def rnd_order_test(df, clf):
    #Need to read in the same *exact* phecode list that was used to train the classifier, in that order
    #This should generally be saved, in order, for use with the classifier
    phe_df = pd.read_csv('/data/ruderferlab/projects/biovu/trainwreck/files/cc_phe_list_unique_ordered.csv', dtype=str)
    phe_list = list(phe_df['phecode'])
    #Now randomize order
    shuffle(phe_list)
    probs = clf.predict_proba(df[phe_list])
    df['rnd_order_case_prb'] = probs[:, 1]
    return roc_auc_score(df['gClin'], df['rnd_order_case_prb'])


#run all three methods n times, using df for grids and input, clf for prediction
def compare_random_methods(df, clf, n):
    biased_aucs = []
    uniform_aucs = []
    rnd_order_aucs = []
    for i in range(n):
        print(n)
        biased_aucs.append(biased_test(df))
        uniform_aucs.append(uniform_prb_test(df))
        rnd_order_aucs.append(rnd_order_test(df, clf))
    print('biased random probabilities mean roc_auc: ', statistics.mean(biased_aucs))
    print(biased_aucs)
    print('uniform random probabilities mean roc_auc: ', statistics.mean(uniform_aucs))
    print(uniform_aucs)
    print('random order mean roc_auc: ', statistics.mean(rnd_order_aucs))
    print(rnd_order_aucs)


if __name__=='__main__':
    print('starting')
    #df contains grids, phecode data, gclin
    df = pd.read_csv(sys.argv[1])
    n = int(sys.argv[2])
    clf = pickle.load(open(sys.argv[3], 'rb'))
    print('read input')
    compare_random_methods(df, clf, n)
