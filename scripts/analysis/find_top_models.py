import sys
import numpy as np
import pandas as pd

if __name__=='__main__':
    df = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    for i in ['0','1','2','3']:
        df['split'+i+'_test_ppv']=df['split'+i+'_test_tp']/(df['split'+i+'_test_tp']+df['split'+i+'_test_fp'])
        df['split'+i+'_test_recall']=df['split'+i+'_test_tp']/(df['split'+i+'_test_tp']+df['split'+i+'_test_fn'])
        df['split'+i+'_test_f1']=2*((df['split'+i+'_test_ppv']*df['split'+i+'_test_recall'])/(df['split'+i+'_test_ppv']+df['split'+i+'_test_recall']))
    df['mean_test_ppv']=df[['split0_test_ppv','split1_test_ppv','split2_test_ppv','split3_test_ppv']].sum(axis=1)/4
    df['mean_test_recall']=df[['split0_test_recall','split1_test_recall','split2_test_recall','split3_test_recall']].sum(axis=1)/4
    df['mean_test_f1']=df[['split0_test_f1','split1_test_f1','split2_test_f1','split3_test_f1']].sum(axis=1)/4
    df.to_csv(out, index=False)
