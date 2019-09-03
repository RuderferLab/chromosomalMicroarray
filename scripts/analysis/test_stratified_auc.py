import sys
import pandas as pd
from sklearn.metrics import roc_auc_score


def stratifyAndTest(df):
    #bin by number of unique phecodes
    ranges = [(n, min(n+10, 200)) for n in range(0, 200, 10)]
    ranges.append((200, max(df['unique_cc_phecode_count'])+1))
    true_aucs = []
    fake_aucs = []
    for b in ranges:
        print(b)
        true_aucs.append(roc_auc_score(df.loc[(df['unique_cc_phecode_count']>=b[0])&(df['unique_cc_phecode_count']<b[1]), 'gClin'], df.loc[(df['unique_cc_phecode_count']>=b[0])&(df['unique_cc_phecode_count']<b[1]), 'true_case_prob']))
        fake_aucs.append(roc_auc_score(df.loc[(df['unique_cc_phecode_count']>=b[0])&(df['unique_cc_phecode_count']<b[1]), 'gClin'], df.loc[(df['unique_cc_phecode_count']>=b[0])&(df['unique_cc_phecode_count']<b[1]), 'random_order_case_prob']))
    res_df = pd.DataFrame()
    res_df['bins']=ranges
    res_df['fake_aucs']=fake_aucs
    res_df['true_aucs']=true_aucs
    return res_df


def stratifyAndTestDeciles(df):
    la = list(range(10))
    df['decile'] = pd.qcut(df['unique_cc_phecode_count'], 10, labels=range(10))
    true_aucs = []
    fake_aucs = []
    for label in la:
        print(label)
        true_aucs.append(roc_auc_score(df.loc[df['decile']==label, 'gClin'], df.loc[df['decile']==label, 'true_case_prob']))
        fake_aucs.append(roc_auc_score(df.loc[df['decile']==label, 'gClin'], df.loc[df['decile']==label, 'random_order_case_prob']))
    res_df = pd.DataFrame()
    res_df['deciles']=la
    res_df['fake_aucs']=fake_aucs
    res_df['true_aucs']=true_aucs
    return res_df

#args: df out
if __name__=='__main__':
    #Read in df containing unique_cc_phecode_count, true_case_prob, gClin, and random_order_case_prob
    df = pd.read_csv(sys.argv[1])
    res=stratifyAndTestDeciles(df)
    print(res)
    res.to_csv(sys.argv[2], index=False)
