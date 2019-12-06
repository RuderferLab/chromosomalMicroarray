import sys
import gc
import pandas as pd
from scipy.sparse import csr_matrix
import dask.dataframe as dd
import match_controls
import time
import cross_validation_pipeline_probabilistic
import create_weight_df

'''
Plan: take all parts of full pipeline, go from start (creaing case control set) to end (selecting best model, predicting on the frequent visitors)
Inputs rough:
    CMA set and data
    All grids in SD with demographic data
    Frequent Visitors set (Could just have a field marking them in the full SD file)
    list of phecodes for everyone
    weights of each phecode across freq_vis
Inputs final:
    CMA file
    demographic data for everyone in SD (freq_vis marked)
    phecode list for everyone in SD
    phecode weights
steps:
    1. Match controls, create markings of case and control in big dataframe with demographics
    2. Create long form dataframe of everyone with phecode counts + demographic info + cc marking + fv marking
    3. Seperate out case control set and provide as input to the pipeline
    4. Select best model from the pipeline, retrain it on the whole CC set, and predict on the frequent visitors set
'''

CENSOR=False

#To censor codes, needs GRID, CODE, and DATE columns
def censor_codes(code_df, cma_df):
    print(code_df.head())
    #Look at only cma codes
    cma_codes = code_df.loc[code_df.GRID.isin(cma_df.GRID)]
    print(cma_codes.head())
    cma_merge = cma_codes.merge(cma_df[['GRID','ENTRY_DATE']], how='left', on='GRID')
    cma_merge.ENTRY_DATE = pd.to_datetime(cma_merge.ENTRY_DATE)
    cma_merge.DATE = pd.to_datetime(cma_merge.DATE)
    print(cma_merge.head())
    #Identify indices to drop
    drop_inds = cma_merge.loc[cma_merge.ENTRY_DATE<=cma_merge.DATE].index
    #Drop them
    return code_df.drop(index=drop_inds)


#python master.py demographic_data phecodes weights CMA fv_out param_out
def main():
    start_time = time.time()
    #Load in all inputs
    #Demographic data for entire SD
    demo_df = pd.read_csv(sys.argv[1], sep='|')
    #Phecodes for entire SD
    phecodes = dd.read_csv(sys.argv[2], dtype=str)
    #Inverse prevalence weights (across frequent visitors) for all phecodes
    weight_df = pd.read_csv(sys.argv[3], dtype={'PHECODE': str})
    #Chromosomal microarray set (individuals who received a genetic test)
    cma_df = pd.read_csv(sys.argv[4]).drop_duplicates(subset='GRID')
    #Location (string) where the frequent visitors case probability predictions should be written to
    fv_out = sys.argv[5]
    #Location (string) where the parameters and cross validation results for all tested models should be written to
    param_out = sys.argv[6]
    cpu_num = int(sys.argv[7])
    cc_out = sys.argv[8]
    probs_out = sys.argv[9]
    print('Loaded args')
    print("--- %s seconds ---" % (time.time() - start_time))
    #Set cc_status (case control status) tag in demo_df to indicate which patients received a CMA
    ##demo_df['CC_STATUS']=0
    ##demo_df.loc[demo_df.GRID.isin(cma_df.GRID),'CC_STATUS']=1
    #Match controls 4:1 with the cma recipients
    print('Beginning to match controls')
    cc_grids = list(match_controls.dask_cc_match(demo_df, cma_df.GRID.unique(), 4))
    print(len(cc_grids))
    print(cma_df.GRID.unique().shape)
    print('matched controls')
    print("--- %s seconds ---" % (time.time() - start_time))
    #Create long df for case control set
    phecodes.columns=['PERSON_ID','GRID', 'CODE','DATE']
    cc_phecodes = phecodes.loc[phecodes.GRID.isin(cc_grids)].compute().reset_index(drop=True)
    print('selected cc_phecodes')
    print("--- %s seconds ---" % (time.time() - start_time))
    if CENSOR:
        cc_phecodes = censor_codes(cc_phecodes, cma_df)
        print('censored cc codes')
        print("--- %s seconds ---" % (time.time() - start_time))
    cc_phecodes_group = cc_phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    long_cc_df = demo_df.loc[demo_df.GRID.isin(cc_grids)].copy().merge(cc_phecodes_group, on="GRID", how="left")
    long_cc_df[cc_phecodes.CODE.unique()] = long_cc_df[cc_phecodes.CODE.unique()].fillna(0).astype(int)
    long_cc_df['CC_STATUS']=0
    long_cc_df.loc[long_cc_df.GRID.isin(cma_df.GRID),'CC_STATUS']=1
    print('Created long cc df')
    print("--- %s seconds ---" % (time.time() - start_time))
    long_cc_df.to_csv(cc_out, index=False)
    #Create long df for fv
    fv_phecodes = phecodes.loc[phecodes.GRID.isin(demo_df.loc[demo_df.FV_STATUS==1,'GRID'])].compute().reset_index(drop=True)
    fv_phecodes_group = fv_phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    long_fv_df = demo_df.loc[demo_df.FV_STATUS==1].copy().merge(fv_phecodes_group, on="GRID", how="left")
    long_fv_df[fv_phecodes.CODE.unique()] = long_fv_df[fv_phecodes.CODE.unique()].fillna(0).astype(int)
    print('created long fv df')
    print("--- %s seconds ---" % (time.time() - start_time))
    ##Get the weight sum (Phenotypic risk score)
    weights = pd.Series(weight_df.WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    print('loaded weights')
    print("--- %s seconds ---" % (time.time() - start_time))
    #drop cogenital anomalies (would be cheating to use them)
    for code in ['758', '758.1', '759', '759.1']:
        if code in long_cc_df:
            long_cc_df[code]=0
        if code in long_fv_df:
            long_fv_df[code]=0
    phe_list = list(weight_df.PHECODE.unique())
    anc_child_dict = create_weight_df.create_ancestry(phe_list)
    leaf_select_cc_df = create_weight_df.leaf_select_codes(long_cc_df, anc_child_dict, phe_list)
    leaf_select_fv_df = create_weight_df.leaf_select_codes(long_fv_df, anc_child_dict, phe_list)
    summed_cc_weight = create_weight_df.get_sums(leaf_select_cc_df, weights)['weight_sum']
    summed_fv_weight = create_weight_df.get_sums(leaf_select_fv_df, weights)['weight_sum']
    #copy over weight sum to long df, since we want all phecodes present for the ml process
    long_cc_df['weight_sum']=summed_cc_weight
    long_fv_df['weight_sum']=summed_fv_weight
    print('summed weights; performed ancestry analysis w/codes')
    print("--- %s seconds ---" % (time.time() - start_time))
    #Run sklearn pipeline for grid search
    phe_list = [x for x in list(weight_df.PHECODE.unique()) if x in long_cc_df.columns]
    phe_list.append('weight_sum')
    #Garbage collect time?
    gc.collect()
    results_df, best_estimator, test_set_probs = cross_validation_pipeline_probabilistic.sklearn_pipeline(long_cc_df[phe_list], long_cc_df['CC_STATUS'].astype(int), cpu_num, 'grid')
    results_df.to_csv(param_out,index=False)
    test_set_probs.to_csv(probs_out, index=False)
    print('pipeline complete')
    print("--- %s seconds ---" % (time.time() - start_time))
    ##retrain best parameters on entire cc set (new problem, same model)
    best_estimator.fit(long_cc_df[phe_list], long_cc_df['CC_STATUS'].astype(int))
    print('estimator retrained')
    print("--- %s seconds ---" % (time.time() - start_time))
    #predict on frequent visitors set (minus the cases and controls)
    fv_df=long_fv_df.loc[~long_fv_df.GRID.isin(cc_grids)].copy()
    fv_df = fv_df.sample(frac=1)
    probs=best_estimator.predict_proba(fv_df[phe_list])
    print('prediction complete')
    print("--- %s seconds ---" % (time.time() - start_time))
    #fv_df['control_prob'] = probs[:, 0]
    fv_df['case_prob'] = probs[:, 1]
    fv_df[['GRID','case_prob']].to_csv(fv_out,index=False)
    print('wrote results')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()
