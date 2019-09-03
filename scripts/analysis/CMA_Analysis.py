import pandas as pd
import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.genmod.families.family as fam
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import chi2_contingency as chi2_c
from assocplots.manhattan import *
#import datashader as ds
#import datashader.utils as utils
#import datashader.transfer_functions as tf

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

def plot_strange_cluster():
    print('started')
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list=[phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('loaded')
    #Create embeddings
    pca = PCA(n_components=50, random_state=42)
    pc_emb = pca.fit_transform(phedf)
    ump = umap.UMAP(metric='euclidean', n_components=10, random_state=42)
    ump_emb = ump.fit_transform(pc_emb)
    print('embedded')
    #create df
    reduced_df = pd.DataFrame(ump_emb, columns = ['UMP-'+str(i+1) for i in range(10)])
    reduced_df['CC_STATUS']=df['CC_STATUS']
    #Create visualization
    sns.set()
    sns.pairplot(reduced_df, hue="CC_STATUS", vars=['UMP-'+str(i+1) for i in range(10)], height=4, markers=['o', 's'], plot_kws=dict(alpha=0.1))
    plt.savefig(out)
    print('graphed')
    #test components
    reduced_df['newcc']=0
    reduced_df.loc[reduced_df['UMP-2']<-12, 'newcc']=1
    df['newcc']=reduced_df['newcc']
    #plot the distribution of this component
    #Age
    df['AGE']= pd.to_datetime(df['BIRTH_DATETIME'].str[:10], format='%Y-%m-%d')
    df['AGE']=(datetime.datetime.now()-df['AGE']).astype('timedelta64[Y]')
    sns.set()
    sns.distplot(df.loc[df['newcc']==0, 'AGE'])
    sns.distplot(df.loc[df['newcc']==1, 'AGE'])
    plt.savefig(out+'_age.png')
    plt.clf()
    #number of phecodes
    sns.distplot(df.loc[df['newcc']==0, 'UNIQUE_PHECODES'])
    sns.distplot(df.loc[df['newcc']==1, 'UNIQUE_PHECODES'])
    plt.savefig(out+'_unique_phecodes.png')
    plt.clf()


def test_umap_one():
    print('started')
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list=[phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('loaded')
    #Create embeddings
    pca = PCA(n_components=50, random_state=42)
    pc_emb = pca.fit_transform(phedf)
    ump = umap.UMAP(metric='euclidean', n_components=10, random_state=42)
    ump_emb = ump.fit_transform(pc_emb)
    print('embedded')
    #create df
    reduced_df = pd.DataFrame(ump_emb, columns = ['UMP-'+str(i+1) for i in range(10)])
    reduced_df['CC_STATUS']=df['CC_STATUS']
    #Create visualization
    sns.set()
    sns.pairplot(reduced_df, hue="CC_STATUS", vars=['UMP-'+str(i+1) for i in range(10)], height=4, markers=['o', 's'], plot_kws=dict(alpha=0.1))
    plt.savefig(out)
    print('graphed')
    #test components
    reduced_df['newcc']=0
    reduced_df.loc[reduced_df['UMP-2']<-12, 'newcc']=1
    df['newcc']=reduced_df['newcc']
    print('opening file')
    out_file = open('files/umap_new_cases_chi_phecode_test_2.csv', 'w')
    out_file.write('phecode,chi2,p,dof,control_neg,case_neg,control_pos,case_pos\n')
    #Run univariate tests using this newcc col
    for phecode in phe_list:
        #Get count of people positive for this phecode in case
        case_pos = df.loc[(df.newcc==1) & (df[phecode]==1)].shape[0]
        #Get negative count in case
        case_neg = df.loc[(df.newcc==1) & (df[phecode]==0)].shape[0]
        #Get positive control
        control_pos = df.loc[(df.newcc==0) & (df[phecode]==1)].shape[0]
        #Get negative control
        control_neg = df.loc[(df.newcc==0) & (df[phecode]==0)].shape[0]
        #Run contingency test
        if case_pos>0 and case_neg>0 and control_pos>0 and control_neg>0:
            res=chi2_c([[control_neg, case_neg],[control_pos, case_pos]])
            #Write results
            out_file.write(','.join([phecode,str(res[0]),str(res[1]),str(res[2]),str(control_neg),str(case_neg),str(control_pos),str(case_pos)]))
            out_file.write('\n')
    out_file.close()
    print('ran phecode tests')
    #Get age
    df['AGE']= pd.to_datetime(df['BIRTH_DATETIME'].str[:10], format='%Y-%m-%d')
    df['AGE']=(datetime.datetime.now()-df['AGE']).astype('timedelta64[Y]')
    #Run same test procedure for covariates, but do regression (?)
    print('running regression')
    mod = smf.glm(formula='newcc ~ AGE + UNIQUE_PHECODES + RACE + GENDER + RECORD_LENGTH_DAYS', data=df, family=fam.Binomial())
    res = mod.fit()
    print(res.summary())



'''
Input:
    components: num_subjects x num_comp+1 dataframe which is transformed into embeddings of original data, along with a column labelled `class` which contains the targets
    out: where to write files to
'''
def abstracted_one_dimension(components, out):
    sns.set()
    components = components.sort_values(by="class").reset_index()
    fig=plt.gcf()
    fig.set_size_inches(20, 16)
    for component in [x for x in list(components.columns.values) if x!='class']:
        sns.scatterplot(x=components.index.values, y=component, hue="class", data=components)
        plt.title(component)
        plt.savefig(out+'_'+component+'.png')
        plt.clf()


#df phecodes out n_comp
def run_one_dim_umap(met):
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    n_comp = sys.argv[4]
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    #Fit umap
    ump = umap.UMAP(metric=met, n_components=int(n_comp))
    embedding = ump.fit_transform(df[phe_list])
    comp_df = pd.DataFrame(embedding, columns=['ump_'+str(i+1) for i in range(int(n_comp))])
    #set class label
    comp_df['class'] = df['CC_STATUS']
    #call abstracted function
    abstracted_one_dimension(comp_df, out)


def principal_component_one_dimension(num_comp):
    sns.set()
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('loaded')
    #Fit and transform pca
    pca = PCA(n_components=num_comp)
    embedding = pca.fit_transform(phedf)
    pc_df = pd.DataFrame(embedding, columns = ['PC-'+str(ind+1) for ind in range(num_comp)])
    pc_df['CC_STATUS'] = df.loc[:,"CC_STATUS"]
    #Plot first num_comp transformed components in one dimension
    #Hue by class
    pc_df = pc_df.sort_values(by="CC_STATUS").reset_index()
    fig = plt.gcf()
    fig.set_size_inches(20, 16)
    for component in range(1,num_comp+1):
        sns.scatterplot(x=pc_df.index.values, y="PC-"+str(component), hue="CC_STATUS", data=pc_df)
        plt.savefig(out+'_PC-'+str(component)+'.png')
        plt.clf()

def top_correlations(should_cutoff, limit):
    corr_df = pd.read_csv(sys.argv[1], index_col=0)
    out = sys.argv[2]
    if should_cutoff:
        corr_df = corr_df.iloc[:limit,:]
    #Sort the dataframe by the correlation pairs
    components = []
    corrs = []
    names = []
    for v in corr_df.index.values:
        #Get top 3 for this component
        top_5 = corr_df.loc[v,:].abs().sort_values(ascending=False).iloc[:5].index.values
        #add values to corresponding lists
        components=components+[v]*5
        names=names+list(top_5)
        corrs=corrs+[corr_df.loc[v, phec] for phec in top_5]
        print(list(top_5))
        print([corr_df.loc[v, phec] for phec in top_5])
    #Create final listing
    print(components)
    print(corrs)
    res = pd.DataFrame([components, names, corrs], index=['component','name','correlation']).transpose()
    res.to_csv(out, index=False)


def heatmap_corrdf():
    #Read df in
    corr_df = pd.read_csv(sys.argv[1], index_col=0)
    out = sys.argv[2]
    #Limit the size to some extent?
    #Create heatmap
    sns.set()
    fig = plt.gcf()
    fig.set_size_inches(50, 20)
    for i in range(int(math.floor(corr_df.shape[1]/100))):
        sns.heatmap(corr_df.iloc[:,i*100:(i+1)*100], vmin=-1, vmax=1, center=0, cmap='RdBu')
        plt.savefig(out+'_'+str(i*100)+'_'+str((i+1)*100)+'.png')
        plt.clf()
    #Now plot remaining section
    sns.heatmap(corr_df.iloc[:,int(math.floor(corr_df.shape[1]/100))*100:], center=0, vmin=0, vmax=1, cmap='RdBu')
    plt.savefig(out+'_'+str(math.floor(corr_df.shape[1]/100))+'_END.png')
    plt.clf()



def run_all_correlation(metric,n_comp):
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('loaded')
    #Get age
    df['AGE']= pd.to_datetime(df['BIRTH_DATETIME'].str[:10], format='%Y-%m-%d')
    df['AGE']=(datetime.datetime.now()-df['AGE']).astype('timedelta64[Y]')
    print('got age')
    #Create location+result
    df['loc_res']=df['location']+' '+df['Result']
    #Run umap
    umap_df = umap_components(metric, n_comp, df, phe_list)
    print('umap done')
    #Run pca
    pca_df, pca = pca_components(df, phe_list, n_comp)
    print('pca done')
    #get umap and pca correlation matrices
    #pca_corr = pd.DataFrame(pca.components_,columns=df[phe_list].columns,index = ['PC-'+str(i+1) for i in range(pca.n_components_)])
    umap_corr_res = cov_correlation_components(umap_df, df, phe_list)
    pca_corr_res = cov_correlation_components(pca_df, df, phe_list)
    print('correlation done')
    #Plot heatmaps and save results
    sns.set()
    fig = plt.gcf()
    fig.set_size_inches(50,20)
    sns.heatmap(umap_corr_res, vmin=-1, vmax=1, center=0, cmap='RdBu')
    plt.savefig(out+'_UMAP.png')
    plt.clf()
    sns.heatmap(pca_corr_res, vmin=-1, vmax=1, center=0, cmap='RdBu')
    plt.savefig(out+'_PCA.png')
    print('figs done')
    plt.clf()


'''
Input: Dataframe which contains each component, as well as the dataframe with the demographic info
'''
def cov_correlation_components(component_df, full_df, phe_list):
    covs = ['RACE', 'GENDER', 'RECORD_LEN', 'AGE', 'CC_STATUS', 'loc_res']
    important_area=full_df[covs]
    for c in ['RACE', 'GENDER', 'CC_STATUS', 'loc_res']:
        important_area.loc[:,c]=important_area[c].astype('category').cat.codes
    #Get correlation
    cres=pd.concat([important_area, component_df], axis=1, keys=['important_area', 'comp_df']).corr().loc['comp_df', 'important_area']
    return cres


def pca_components(df, phe_list, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(df[phe_list])
    #res= pd.DataFrame(pca.components_,columns=df[phe_list].columns,index = ['PC-'+str(i) for i in range(df, phe_list, pca.n_components_)])
    return (pd.DataFrame(pca.transform(df[phe_list]), columns=['PC-'+str(i+1) for i in range(n_comp)]), pca)


'''
Input: String specifying metric and int specifying number of components.
Output: Dataframe containing each component, labelled from 1 to num_components as ump_n, where n is the number. Also the dataframe with demo info
'''
def umap_components(met, n_comp, df, phe_list):
    ump = umap.UMAP(metric=met, n_components=n_comp)
    embedding = ump.fit_transform(df[phe_list])
    umap_df = pd.DataFrame(embedding, columns=['ump_'+str(i+1) for i in range(n_comp)])
    #Return transformed dataframe
    return umap_df


#BIRTH_DATETIME,RACE,GENDER,RECORD_LEN
def component_importance(met):
    #sns.set()
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('all loaded')
    ump = umap.UMAP(metric=met, n_components=20)
    embedding = ump.fit_transform(df[phe_list])
    print('umap done')
    #get the feature importance for umap
    important_area = df[phe_list+['RACE','GENDER','RECORD_LEN']]
    important_area['RACE'] = important_area['RACE'].astype('category').cat.codes
    important_area['GENDER'] = important_area['GENDER'].astype('category').cat.codes
    #ures=important_area.corrwith(pd.DataFrame(embedding, columns=['ump_'+str(i+1) for i in range(20)]))
    umap_df=pd.DataFrame(embedding, columns=['ump_'+str(i+1) for i in range(20)])
    ures=pd.concat([important_area, umap_df], axis=1, keys=['important_area', 'umap_df']).corr().loc['umap_df', 'important_area']
    print(ures)
    ures.to_csv(out+'_umap_importance.csv')
    #Do pca
    pca = PCA()
    pca.fit(df[phe_list])
    res= pd.DataFrame(pca.components_,columns=df[phe_list].columns,index = ['PC-'+str(i) for i in range(pca.n_components_)])
    print(res)
    res.to_csv(out+'_pca_importance.csv')


def pca_umap_compare(met):
    sns.set()
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    ##Only consider results which are gain, loss, normal, or 0 (0 is a control)
    ##df = df.loc[df.Result.isin(['Gain','Loss','Normal','0'])]
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    #binarize
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    print('all loaded')
    ##Get top 10 cnv's in order to set them not to be grayed out
    ##top_locations = df.loc[df.Result.isin(['Gain','Loss'])].groupby(['Result','location']).count().GRID.sort_values(ascending=False).reset_index()
    #UMAP on whole dataset
    ump = umap.UMAP(metric=met)
    embedding = ump.fit_transform(df[phe_list])
    print('umap embedding complete')

    #Vizualization with umap -- set class as case control
    viz_df = pd.DataFrame(embedding, columns=('ump_1', 'ump_2'))
    viz_df['class']=pd.Series(df.CC_STATUS.astype(int).astype(str), dtype='category')
    print(viz_df['class'].unique())
    fig = plt.gcf()
    fig.set_size_inches( 16, 10)
    #sns.pairplot(x_vars=['ump_1'], y_vars=['ump_2'], data=viz_df, hue="class", size=8, plot_kws={'alpha':0.25})
    sns.scatterplot(x='ump_1', y='ump_2', data=viz_df, hue="class", alpha=0.35)
    plt.title("Case Control dataset, UMAP embedding: "+met)
    plt.savefig(out+'_UMAP_Case_Control_'+met+'.png')
    print('fig 1 done')
    plt.clf()

    #Same umap embedding -- set class as greyout vs top 10 cnv
    #Get top non-control/non-normal cnvs
    df['concat'] = df['location']+'_'+df['Result']
    loc_res_counts = df.loc[~df.location.isin(['0','-'])].groupby('concat').count().GRID.sort_values(ascending=False).reset_index()
    top_locs = list(loc_res_counts.loc[loc_res_counts.GRID>4].concat)
    #Create column with "OTHER" designation for labels outside of top_locs
    df['filtered_concat'] = 'OTHER'
    df.loc[df.concat.isin(top_locs),"filtered_concat"] = df.concat
    ###palette = dict(zip(df.concat.unique(), ['gray']*len(df.concat.unique())))
    ##Change all elements which are not in the selected (colored) cnv pairs of result/location
    
    #Create colormap for values of filtered_concat
    cmap = sns.color_palette('Spectral', len(top_locs))
    palette = dict(zip(top_locs, cmap))
    palette["OTHER"]='gray'
    print(palette)
    ##counter = 0
    ##for l in df.concat.unique():
    ##    if l in top_locs:
    ##        palette.update({l:cmap[counter]})
    ##        counter+=1
    #Create the actual visualization -- reset the class column with new classes
    viz_df['class'] = df['filtered_concat']
    sns.scatterplot(x='ump_1', y='ump_2', data=viz_df.loc[~viz_df['class'].isin(top_locs),:], hue="class", palette=palette, alpha=0.25)
    sns.scatterplot(x='ump_1', y='ump_2', data=viz_df.loc[viz_df['class'].isin(top_locs),:], hue="class", palette=palette, alpha=0.65)
    #sns.pairplot(x_vars=['ump_1'], y_vars=['ump_2'], data=viz_df.loc[~viz_df['class'].isin(top_locs),:], hue="class", palette=palette, size=8, plot_kws={'alpha':0.25})
    #sns.pairplot(x_vars=['ump_1'], y_vars=['ump_2'], data=viz_df.loc[viz_df['class'].isin(top_locs),:], hue="class", palette=palette, size=8, plot_kws={'alpha':0.35})
    plt.title('Case Control Dataset colored by top CNVs, UMAP: '+met)
    plt.savefig(out+'_UMAP_CNVs_over_4_'+met+'.png')
    plt.clf()
    print('fig 2 done')
    #Next do PCA
    #Get proportion of explained variance for pca
    pca = PCA()
    pca.fit(df[phe_list])
    print('first pc fit done')
    #Plot it
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig(out+'_PCA_Explained_variance.png')
    plt.clf()
    print('explained variance done')
    #fit and transform for PCA on two components
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(df[phe_list])
    print('PC fit transform done')
    #Plot the case vs control on the top 2 PCs
    viz_df = pd.DataFrame(pca_embedding, columns=('pc_1', 'pc_2'))
    viz_df['class']=pd.Series(df.CC_STATUS.astype(int).astype(str), dtype='category')
    sns.scatterplot(x='pc_1', y='pc_2', data=viz_df, hue="class", alpha=0.35)
    #sns.pairplot(x_vars=['pc_1'], y_vars=['pc_2'], data=viz_df, hue="class", size=8, plot_kws={'alpha':0.25})
    plt.title("Case control dataset: PCA embedding")
    plt.savefig(out+'_case_control_PCA.png')
    plt.clf()
    print('fig 3 done')
    #Plot the greyed out everything but top CNVs for top 2 PCs
    viz_df['class'] = df['filtered_concat']
    sns.scatterplot(x='pc_1', y='pc_2', data=viz_df.loc[~viz_df['class'].isin(top_locs),:], hue="class", palette=palette, alpha=0.25)
    sns.scatterplot(x='pc_1', y='pc_2', data=viz_df.loc[viz_df['class'].isin(top_locs),:], hue="class", palette=palette, alpha=0.65)
    #sns.pairplot(x_vars=['pc_1'], y_vars=['pc_2'], data=viz_df.loc[~viz_df['class'].isin(top_locs),:], hue="class", palette=palette, size=8, plot_kws={'alpha':0.25})
    #sns.pairplot(x_vars=['pc_1'], y_vars=['pc_2'], data=viz_df.loc[viz_df['class'].isin(top_locs),:], hue="class", palette=palette, size=8, plot_kws={'alpha':0.35})
    plt.title('Case Control Dataset colored by top CNVs: PCA')
    plt.savefig(out+'_PCA_top_CNVs.png')
    plt.clf()
    print('fig 4 done')

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


#def CNV_plots_all(split, mets):
#    #Take the 

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
    #CNV_colored_plots_smallset(True, ['jaccard', 'cosine', 'hamming', 'dice', 'correlation', 'russellrao', 'euclidean', 'yule'])
    #pca_umap_compare('jaccard')
    #component_importance('jaccard')
    #heatmap_corrdf()
    #top_correlations(False, 20)
    #principal_component_one_dimension(10)
    #run_one_dim_umap('jaccard')
    #run_all_correlation('jaccard', 20)
    #test_umap_one()
    #plot_strange_cluster()
