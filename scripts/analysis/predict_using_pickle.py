import pandas as pd
import sys
import pickle


def main():
    df = pd.read_csv(sys.argv[1])
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    clf =  pickle.load(open(sys.argv[3], 'rb'))
    phe_list = list(phecodes.PHECODE.unique())
    #predict phe_list of df using clf
    predicted_probs = clf.predict_proba(df[phe_list])
    df['control_prob'] = predicted_probs[:, 0]
    df['case_prob'] = predicted_probs[:, 1]
    #Save predictions to file

if __name__=='__main__':
    main()
