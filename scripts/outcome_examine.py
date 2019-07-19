import sys
import datetime as DT
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import numpy as np

'''
input_var: Single string column name in df.
outcome_vars: List of string column names to test and output the relationship of with input_var (also must be in df)
df: A dataframe containing columns corresponding to the above variables.
'''
def test_relationships(input_var, outcome_vars, df):
    for var in outcome_vars:
        pr = pearsonr(df[input_var], df[var])
        sr = spearmanr(df[input_var], df[var])
        plt.plot(df[input_var], df[var], alpha=0.3)
        plt.savefig(str(var)+'_'+str(input_var)+"scatter.png")
        plt.clf()

def glm_test(df):
    now = pd.Timestamp(DT.datetime.now())
    print("Loaded data")
    print(df.shape)
    df.dropna(inplace=True,subset=['MOST_RECENT_EVENT','BIRTH_DATETIME'])
    df['BIRTH_DATETIME']=df['BIRTH_DATETIME'].str[:-2]
    df['MOST_RECENT_EVENT']=df['MOST_RECENT_EVENT'].str[:-2]
    print(df.shape)
    #Get time since last mre
    df["MRE"]=pd.to_datetime(df['MOST_RECENT_EVENT'], format="%Y-%m-%d %H:%M:%S")
    df['time_since_MRE']=(now - df['MRE']).astype('<m8[Y]')
    #Get age from dob
    df["dob"]=pd.to_datetime(df['BIRTH_DATETIME'], format="%Y-%m-%d %H:%M:%S")
    df['age']=(now - df['dob']).astype('<m8[Y]')
    #Change abnormal_normal NaNs to "not_tested"
    df.loc[(df['normal_abnormal']!="Abnormal")&(df['normal_abnormal']!="Normal"), "normal_abnormal"]="not_tested"
    print("Converted Data")
    #Get glm
    results=sm.formula.glm(formula='weight_sum ~ GENDER + age + RACE + normal_abnormal + GENETIC_CPT_STATUS + gClin + cc_status + time_since_MRE', data=df, missing='drop').fit()
    print(results.summary())
    print(np.exp(results.params))
    return df

def modify_scores(df,out):
    #df['age_adjusted'] = df['weight_sum']/df['age']
    df['age_adjusted'] = np.where(df['age'] > 0, df['weight_sum']/df['age'], df['weight_sum'])
    df[['GRID','age_adjusted','weight_sum','time_since_MRE', 'cc_status', 'gClin', 'GENETIC_CPT_STATUS']].to_csv(out, index=False)

if __name__=='__main__':
    df = pd.read_csv(sys.argv[1], index_col=0)
    df_new = glm_test(df)
    modify_scores(df_new, sys.argv[2])
