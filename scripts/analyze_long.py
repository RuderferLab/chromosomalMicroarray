from pandas import read_csv, DataFrame
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def cut_string(s):
    return s[:25]

def combine_string(s):
    if 'delay' in s.lower() or 'developmental' in s.lower() or 'lack of expected normal' in s.lower():
        return 'developmental delay'
    elif 'congenital' in s.lower():
        return 'congenital anomalies'
    else:
        return s

def biclass(s):
    if s != "Normal":
        return "Abnormal"
    else:
        return s

def kb_transform(x):
    if 'mb' in x.lower():
        return 1000*float(x[:-2])
    elif 'kb' in x.lower():
        return float(x[:-2])
    else:
        return x

def age_res():

    markers = {'Gain':'P', 'Loss':'s', 'Normal':'o'}

    df =read_csv(sys.argv[1])

    #Get locations which appear more than 3 times in the dataset

    sizes = DataFrame(df.groupby("location").size()).reset_index()

    print(sizes)

    sizes = sizes.loc[sizes[0]>3]
    
    print(sizes)

    sizes = sizes.loc[sizes.location!="-"]

    fil = df.merge(sizes, on="location", how="inner")

    #fil=fil.loc[fil['Result'].isin(['Gain', 'Loss'])]

    #fig, ax = plt.subplots(figsize=(21,7))
    #sns.violinplot(x="location",y="age", hue="Result", data=fil, split=True)
    #plt.savefig(sys.argv[2])

    fig, ax = plt.subplots(figsize=(21,7))
    #Plot age by location
    sns.swarmplot(x="location",y="age",data=fil)#, marker=markers[res_type])
    plt.savefig(sys.argv[2])
    plt.clf()

def age_res_type():
    df = read_csv(sys.argv[1])
    #Filter to abnormal vs normal
    df['Binary-Result'] = df['Result'].apply(biclass)
    
    #Change unclear results to special category
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result'])

    #Plot
    #sns.boxplot(x="Tri-Result", y="age", data=df)
    sns.swarmplot(x="Tri-Result", y="age", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def phe_res_type():
    df = read_csv(sys.argv[1])
    #Filter to abnormal vs normal
    df['Binary-Result'] = df['Result'].apply(biclass)

    #Change unclear results to special category
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result'])
    print(df.head())
    df['phecode_sum'] = df.drop(['NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','unclear','dob','age','years', 'Tri-Result', 'Binary-Result'], axis=1).sum(axis=1)

    #Plot
    sns.swarmplot(x="Tri-Result", y="phecode_sum", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()


def age_ind():

    df = read_csv(sys.argv[1])

    inds = DataFrame(df.groupby("indication").size()).reset_index()

    #print(inds)

    inds = inds.loc[inds[0]>10]

    #print(inds)

    fil = df.merge(inds, on="indication", how="inner")
    
    fil['indication'] = fil['indication'].apply(combine_string)

    fig, ax = plt.subplots(figsize=(24,7))
    sns.swarmplot(x="indication", y="age", data=fil)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.savefig(sys.argv[2])
    plt.clf()

def age_size():
    df = read_csv(sys.argv[1])

    df = df.loc[df['size']!='-']

    #df=df.loc[df['size']!='entire chromosome']

    df = df.dropna()

    df['kb'] = df['size'].apply(kb_transform)

    df['kb'] = np.where((df['kb']=='entire chromosome') & (df['location']=='X chromosome'), 156040.8, df['kb'])
    df['kb'] = np.where((df['kb']=='entire chromosome') & (df['location']=='chromosome 21'), 46709, df['kb'])
    df['kb'] = np.where((df['kb']=='entire chromosome') & (df['location']=='Y chromosome'), 57227.4, df['kb'])

    float_df = DataFrame(df[['kb','age']], dtype=float)

    float_df = float_df.loc[float_df['kb']<20000]

    sns.regplot(x='kb', y='age', data=float_df)
    plt.savefig(sys.argv[2])
    plt.clf()

if __name__ == "__main__":
    #age_ind()
    age_res()
    #age_res_type()
    #phe_res_type()
    #age_size()
