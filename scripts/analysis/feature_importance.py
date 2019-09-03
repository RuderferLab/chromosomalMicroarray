import sys
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

#Input: full_df phecode_weights
def lasso_test():
    df = pd.read_csv(sys.argv[1])
    weights = pd.read_csv(sys.argv[2])
    #print(df.head())
    #print(weights.head())
    cols = [str(x) for x in weights['PHECODE'] if str(x) in df.columns]
    #print("!")
    #print(cols)
    phe_cols = df[cols]
    targets = df["cc_status"]
    #print(phe_cols.head())
    #print(targets.head())
    for alp in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print("Alpha equal to: "+str(alp))
        clf = linear_model.Lasso(alpha=alp)
        clf.fit(phe_cols, targets)
        zipped=list(zip(clf.coef_, cols))
        zipped_gz=[z for z in zipped if z[0]>0]
        print(zipped_gz)
        print(clf.intercept_)

def corr():
    df = pd.read_csv(sys.argv[1])
    weights = pd.read_csv(sys.argv[2])
    cols = [str(x) for x in weights['PHECODE'] if str(x) in df.columns]
    phe_cols = df[cols]
    print("plotting heatmap")
    corrs = phe_cols.corr()
    corrs.to_csv(sys.argv[3])
    #sns.heatmap(corrs)
    #print(phe_cols.corr())
    #plt.matshow(phe_cols.corr())
    #plt.savefig(sys.argv[3])


def main():
    corr()
    #lasso_test()

if __name__=='__main__':
    main()
