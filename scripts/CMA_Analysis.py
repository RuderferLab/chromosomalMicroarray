import pandas as pd
import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.genmod.families.family as fam
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import chi2_contingency as chi2_c
from assocplots.manhattan import *
import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf

def dconvert(d):
    return datetime.datetime.strptime(d,"%Y-%m-%d")

def ddays(d):
    return int(round(d.days/365))

def binarize_normal(r):
    if r!="Normal":
        return 1
    else:
        return 0

def bin_ages(df, num_bins, mod):
    thresholds = dict()
    for t in range(num_bins):
        thresholds[(t+1)*mod] = df[(df['age']<mod*(t+1)) & (df['age']>=mod*t)].shape[0]
    return thresholds

'''
Input:
    df: Full dataframe
    pl: Columns considered (must be numeric)
    name: Name for count column
'''
def get_phe_counts(df, pl, name):
    phe_df = df.loc[:, pl].astype(int)
    #Binarize the counts (Doesn't matter how many phecodes of a certain type someone has, it is only presence/absence)
    phe_df[phe_df>0] = 1
    #Get sums for each phecode (count of indvs with a phecode
    phe_sums = phe_df.apply(sum)
    #Reset series and label columns so we can easily save it
    sum_df = phe_sums.reset_index()
    sum_df.columns=['PHECODE', name]
    return sum_df



def CNV_colored_plots_smallset(split, mets):
    df = pd.read_csv(sys.argv[1], dtype={'phecode':str, 'location':str, 'Result':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print(df[phe_list].head())
    top_locations = None
    #if only_gln:
    #    #Only consider gains, losses, and normal results
    if split:
        #Consider gains and losses within an abnormal result seperately
        top_locations = df.loc[df.Result.isin(['Gain','Loss'])].groupby(['Result','location']).count().GRID.sort_values(ascending=False).reset_index()
    #Umap visualization
    locs = [top_locations.iloc[l].location for l in range(10)]
    res = [top_locations.iloc[l].Result for l in range(10)]
    colors = [ '#9e0142'
            '#d8434e'
            '#f67a49'
            '#fdbf6f'
            '#feeda1'
            '#f1f9a9'
            '#bfe5a0'
            '#74c7a5'
            '#378ebb'
            '#5e4fa2']
    #Put together all information
    pal = list(zip(locs,res,colors))


    classes = [l+'_'+r for l,r in zip(locs,res)]
    

    df['concat'] = df.location+'_'+df.Result
    df = df.loc[df['concat'].isin(classes)]
    df=df.reset_index()
    print('beginning umap')
    for met in mets:
        print(met)
        ump = umap.UMAP(metric=met)
        embedding = ump.fit_transform(df[phe_list])
        viz_df = pd.DataFrame(embedding, columns=('x', 'y'))
        viz_df['class']=pd.Series(df['concat'], dtype='category')

        color_key = {l+r:c for l,r,c in pal}

        sns.pairplot(x_vars=['x'], y_vars=['y'], data=viz_df, hue="class", size=6)
        plt.title('Locations in UMAP: Small set')
        plt.savefig(sys.argv[3]+'_'+met+'.png')
        plt.clf()

        #calculate centroids
        centroid_df = viz_df.groupby('class').mean().reset_index()
        sns.pairplot(x_vars=['x'], y_vars=['y'], data=centroid_df, hue='class', size=6)
        plt.title('Centroids of UMAP embedded CNVs: Small Set')
        plt.savefig(sys.argv[3]+'_'+met+'_centroids.png')
        plt.clf()


def CNV_plots_all(split, mets):
    #Take the 

'''
Uses phecodes in dataframe to create principal components/UMAP and see if they cluster well
'''
def CNV_colored_plots():
    df = pd.read_csv(sys.argv[1], dtype={'phecode':str, 'location':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    #Umap visualization
    print('beginning umap')
    ump = umap.UMAP()
    embedding = ump.fit_transform(df[phe_list])
    viz_df = pd.DataFrame(embedding, columns=('x', 'y'))
    viz_df['class']=df.location#pd.Series(df['location'].values, dtype='category')

    pal = [('2q13', '#9e0142'),
            ('22q11.21', '#d8434e'),
            ('16p11.2', '#f67a49'),
            ('15q11.2', '#fdbf6f'),
            ('17p12', '#feeda1'),
            ('X chromosome', '#f1f9a9'),
            ('7q11.23', '#bfe5a0'),
            ('15q11.2q13.1', '#74c7a5'),
            ('1q21.1', '#378ebb'),
            ('chromosome 21', '#5e4fa2')]

    color_key = {d:c for d,c in pal}

    #Restrict df to specific most common cnvs (top 10)
    viz_df=viz_df.loc[viz_df['class'].isin(list(zip(*pal))[0])]
    print(viz_df.head())
    viz_df['class']=pd.Series(viz_df['class'], dtype='category')
    print(viz_df.head())

    print('Selected subset of top 10 most common cnvs')

    #Creating image
    #cvs = ds.Canvas(plot_width=400, plot_height=400)
    #agg = cvs.points(viz_df, 'x', 'y', ds.count_cat('class'))
    #img = tf.shade(agg, color_key=color_key, how='eq_hist')
    
    #utils.export_image(img, filename='cnv_shade_umap', background='black')

    #image = plt.imread('cnv_shade_umap.png')
    #fig, ax = plt.subplots(figsize=(6, 6))
    sns.pairplot(x_vars=['x'], y_vars=['y'], data=viz_df, hue="class", size=6)
    #plt.setp(ax,xticks=[],yticks=[])
    plt.title('Locations in UMAP')
    plt.savefig(sys.argv[3])
    plt.clf()


def plot_phewas_manhattan():
    df = pd.read_csv(sys.argv[1], dtype={'phecode':str})
    phewas_category = pd.read_table(sys.argv[2], dtype=str)
    phewas_category = phewas_category.rename(columns={'phewas_code':'phecode'}).loc[:,["phecode","category"]].drop_duplicates()
    merged = df.merge(phewas_category, how='left', on='phecode')
    merged = merged.loc[merged.p>0,:]
    manhattan(merged.p, merged.phecode.astype(float), merged.category, '', xlabel="category")
    plt.title("Case vs Control: Chi-Squared")
    plt.savefig(sys.argv[3])
    plt.clf()


def phewas_upper():
    cc_df = pd.read_csv(sys.argv[1], dtype=str)
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    norm_df = cc_df.loc[cc_df.CC_STATUS=="1.0",:]
    norm_df['NORM_STATUS'] = norm_df["Result"].apply(binarize_normal)
    print(norm_df.head())
    for params in [(cc_df.loc[cc_df.CC_STATUS=="0.0",:],cc_df.loc[cc_df.CC_STATUS=="1.0",:], 'CONTROL', 'CASE', 'cc', cc_df),(norm_df.loc[norm_df.NORM_STATUS==0,:],norm_df.loc[norm_df.NORM_STATUS==1,:], 'NORMAL', 'ABNORMAL', 'norm', norm_df)]:
        phewas_chi_abstract(params[5], phecodes, out+params[4]+'.csv', [params[0], params[1]], [params[2], params[3]])

def phewas_chi_abstract(df, phecodes, out, pairs, name):
    out = open(out, 'w')
    #Identify list of unique phecodes which are also column headers
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    res_df = None
    for pair in [(pairs[0], name[0]+'_COUNT'), (pairs[1], name[1]+'_COUNT')]:
        if res_df is not None:
            res_df = res_df.merge(get_phe_counts(pair[0], phe_list, pair[1]), how='inner', on='PHECODE')
        else:
            res_df = get_phe_counts(pair[0], phe_list, pair[1])
    #Get totals for [controls, cases]
    totals = [pairs[0].shape[0],pairs[1].shape[0]]
    #Create contingency tables and run chi squared test for each phecode
    res_df=res_df.set_index(res_df.PHECODE)
    res_df=res_df.drop('PHECODE',axis=1)
    out.write('phecode,chi2,p,dof,'+name[0]+'_neg,'+name[1]+'_neg,'+name[0]+'_pos,'+name[1]+'_pos\n')
    for phecode in phe_list:
        phecode_counts = list(res_df.loc[phecode])
        #Use phecode counts (case number) alongside totals (both in format of [control#, case#]) to get the count of presence and absence for each phecode
        table = [[total-case for total, case in zip(totals, phecode_counts)], phecode_counts]
        #Calculate the chi2 results
        try:
            if table[1][0]>4 and table[1][1]>4:
                res=chi2_c(table)
                conc = [phecode,str(res[0]), str(res[1]),str(res[2]),str(table[0][0]),str(table[0][1]), str(table[1][0]),str(table[1][1])]
                out.write(','.join(conc)+'\n')
                #if res[1]<=0.0001:
                #    print("~~~~~~~~~")
                #    print(phecode)
                #    print(res)
                #    print(table)
                #    print("~~~~~~~~~")
        except:
            print("The following phecode had invalid expected frequencies for chi2 contingency test: "+phecode)
            print("The table for the above phecode is here: ")
            print(table)
    out.close()



def phewas_chi():
    cc_df = pd.read_csv(sys.argv[1], dtype=str)
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = open(sys.argv[3], 'w')
    #Identify list of unique phecodes which are also column headers
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in cc_df]
    #phe_list.append('CC_STATUS')
    df = None
    #Get sums for each phecode, seperated by case and control
    for pair in [(cc_df.loc[cc_df.CC_STATUS=="0.0",:], 'CONTROL_COUNT'), (cc_df.loc[cc_df.CC_STATUS=="1.0",:],'CASE_COUNT')]:
        if df is not None:
            df = df.merge(get_phe_counts(pair[0], phe_list, pair[1]), how='inner', on='PHECODE')
        else:
            df = get_phe_counts(pair[0], phe_list, pair[1])
    #Get totals for [controls, cases]
    totals = [cc_df.loc[cc_df.CC_STATUS=="0.0",:].shape[0],cc_df.loc[cc_df.CC_STATUS=="1.0",:].shape[0]]
    #Create contingency tables and run chi squared test for each phecode
    df=df.set_index(df.PHECODE)
    df=df.drop('PHECODE',axis=1)
    out.write('phecode,chi2,p,dof,control_neg,case_neg,control_pos,case_pos\n')
    for phecode in phe_list:
        phecode_counts = list(df.loc[phecode])
        #Use phecode counts (case number) alongside totals (both in format of [control#, case#]) to get the count of presence and absence for each phecode
        table = [[total-case for total, case in zip(totals, phecode_counts)], phecode_counts]
        #Calculate the chi2 results
        try:
            if table[1][0]>4 and table[1][1]>4:
                res=chi2_c(table)
                conc = [phecode,str(res[0]), str(res[1]),str(res[2]),str(table[0][0]),str(table[0][1]), str(table[1][0]),str(table[1][1])]
                out.write(','.join(conc)+'\n')
                if res[1]<=0.0001:
                    print("~~~~~~~~~")
                    print(phecode)
                    print(res)
                    print(table)
                    print("~~~~~~~~~")
        except:
            print("The following phecode had invalid expected frequencies for chi2 contingency test: "+phecode)
            print("The table for the above phecode is here: ")
            print(table)
    out.close()


def phecode_table():
    cc_df = pd.read_csv(sys.argv[1], dtype=str)
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    #Identify list of unique phecodes which are also column headers
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in cc_df]
    #Use function to get list of dfs for all subsets of interest
    df = None
    for pair in [(cc_df, 'FULL_CC_COUNT'), (cc_df.loc[cc_df.CC_STATUS=="0.0",:], 'CONTROL_COUNT'), (cc_df.loc[cc_df.CC_STATUS=="1.0",:],'CASE_COUNT')]:
        print(pair[0].head())
        if df is not None:
            df = df.merge(get_phe_counts(pair[0], phe_list, pair[1]), how='inner', on='PHECODE')
        else:
            df = get_phe_counts(pair[0], phe_list, pair[1])
    print(df.head())
    df.to_csv(out, index=False)



def sex_table():
    cc_df = pd.read_csv(sys.argv[1])
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    print(cc_df.groupby(["GENDER", "CC_STATUS"]).GRID.count())
    norm_df = cc_df.loc[cc_df.CC_STATUS==1,]
    norm_df['normality_status'] = norm_df["Result"].apply(binarize_normal)
    print(norm_df.groupby(["GENDER", "normality_status"]).GRID.count())


def ethn_table():
    cc_df = pd.read_csv(sys.argv[1])
    #out = sys.argv[2]
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    print(cc_df.groupby(["RACE", "CC_STATUS"]).GRID.count())
    norm_df = cc_df.loc[cc_df.CC_STATUS==1,]
    norm_df['normality_status'] = norm_df["Result"].apply(binarize_normal)
    print(norm_df.groupby(["RACE", "normality_status"]).GRID.count())



def age_table():
    cc_df = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    cc_df['age'] = datetime.datetime.now() - cc_df["BIRTH_DATETIME"].str[:10].apply(dconvert)
    cc_df['age'] = cc_df['age'].apply(ddays)
    #Plot age distribution for case vs control
    sns.distplot(cc_df.loc[cc_df.CC_STATUS==0, 'age'], label="Age of Controls")
    sns.distplot(cc_df.loc[cc_df.CC_STATUS==1, 'age'], label="Age of Cases")
    plt.title("Age: Case vs Control")
    plt.legend()
    plt.savefig(out+"_case_vs_control.png")
    plt.clf()
    #Plot age distribution for normal vs abnormal
    norm_df = cc_df.loc[cc_df.CC_STATUS==1,]
    norm_df['normality_status'] = norm_df["Result"].apply(binarize_normal)
    sns.distplot(norm_df.loc[norm_df.normality_status==0, 'age'], label="Age of Normal")
    sns.distplot(norm_df.loc[norm_df.normality_status==1, 'age'], label="Age of Abnormal")
    plt.title("Age: Normal vs Abnormal")
    plt.legend()
    plt.savefig(out+"_norm_vs_abnorm.png")
    #Table with age categories
    thresholds = bin_ages(cc_df, 10, 10)
    print("Full case control:")
    print(thresholds.items())
    print("Full cc size:")
    print(cc_df.shape[0])
    print("Cases only:")
    case_threshs = bin_ages(cc_df.loc[cc_df.CC_STATUS==1], 10, 10)
    print(case_threshs.items())
    print("Case count")
    print(cc_df.loc[cc_df.CC_STATUS==1].shape[0])
    print("Controls only:")
    control_threshs = bin_ages(cc_df.loc[cc_df.CC_STATUS==0], 10, 10)
    print(control_threshs.items())
    print("Control count:")
    print(cc_df.loc[cc_df.CC_STATUS==0].shape[0])
    print("Abnormals only")
    abnorms = bin_ages(norm_df.loc[norm_df.normality_status==1], 10, 10)
    print(abnorms.items())
    print("Abnormal count:")
    print(norm_df.loc[norm_df.normality_status==1].shape[0])
    print("Normals only")
    norms = bin_ages(norm_df.loc[norm_df.normality_status==0], 10, 10)
    print(norms.items())
    print("Normal count:")
    print(norm_df.loc[norm_df.normality_status==0].shape[0])


def cma_table():
    cma = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    #Numbers for each CNV/Result pair
    loc_res_table = cma.groupby(by=['location', 'Result'], as_index=False).GRID.count()
    #CNV result numbers with only gain/loss/normal/other, in wide format
    cma['Result_gl'] = cma['Result']
    cma.loc[(cma.Result!="Normal") & (cma.Result!="Gain") & (cma.Result!="Loss"), "Result_gl"]="Other"
    gln_wide = cma.groupby(by=['location', 'Result_gl'], as_index=False).GRID.count().pivot(index="location", columns="Result_gl", values="GRID")
    #Get most frequent reasons for CMA (?)
    ind_table = cma.groupby(by=['indication'], as_index=False).GRID.count().sort_values(by='GRID', ascending=False)
    #Write output
    ind_table.to_csv(out+'ind_table.csv', index=False)
    loc_res_table.to_csv(out+'loc_res_table.csv', index=False)
    gln_wide.to_csv(out+'gln_wide.csv')


def covariate_analysis():
    cc_df = pd.read_csv(sys.argv[1])
    cc_df = cc_df.drop(cc_df[cc_df.BIRTH_DATETIME=='0'].index)
    #Compare sex, age, ethnicity, record_length, and most recent event
    #Get age
    cc_df['age'] = datetime.datetime.now() - cc_df["BIRTH_DATETIME"].str[:10].apply(dconvert)
    cc_df['age'] = cc_df['age'].apply(ddays)
    #Between Case and Control status
    all_res = smf.glm(formula="CC_STATUS ~ weight_sum + RACE + GENDER + age + RECORD_LEN + GENDER*age + age*RECORD_LEN", data=cc_df, family=fam.Binomial()).fit()
    print("Results for Case/control data:")
    print(all_res.summary())
    norm_df = cc_df.loc[cc_df.CC_STATUS==1]
    print(cc_df.shape)
    print(norm_df.shape)
    norm_df['normality_status'] = norm_df["Result"].apply(binarize_normal)
    normality_res = smf.glm(formula="normality_status ~ weight_sum + RACE + GENDER + age + RECORD_LEN + GENDER*age + age*RECORD_LEN", data=norm_df, family=fam.Binomial()).fit()
    print("Results for normal/abnormal data:")
    print(normality_res.summary())



if __name__=="__main__":
    #cma_table()
    #covariate_analysis()
    #age_table()
    #ethn_table()
    #sex_table()
    #phecode_table()
    #phewas_upper()
    #plot_phewas_manhattan()
    CNV_colored_plots_smallset(True, ['jaccard', 'cosine', 'hamming', 'dice', 'correlation', 'russellrao', 'euclidean', 'yule'])
