import sys
import gc
import pandas as pd
import match_controls
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

CALIBRATE=False
CENSOR=True

#To censor codes, needs GRID, CODE, and DATE columns
def censor_codes(code_df, cma_df):
    #Look at only cma codes
    cma_codes = code_df.loc[code_df.GRID.isin(cma_df.GRID)]
    cma_merge = cma_codes.merge(cma_df[['GRID','ENTRY_DATE']], how='left', on='GRID')
    cma_merge.ENTRY_DATE = pd.to_datetime(cma_merge.ENTRY_DATE)
    cma_merge.DATE = pd.to_datetime(cma_merge.DATE)
    #Identify indices to drop
    drop_inds = cma_merge.loc[cma_merge.ENTRY_DATE<=cma_merge.DATE].index
    #Drop them
    return code_df.drop(drop_inds)


#python master.py demographic_data phecodes weights CMA fv_out param_out
def main():
    #Load in all inputs
    #Demographic data for entire SD
    demo_df = pd.read_csv(sys.argv[1], sep='|')
    #Phecodes for entire SD
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    #Inverse prevalence weights (across frequent visitors) for all phecodes
    weight_df = pd.read_csv(sys.argv[3], dtype={'PHECODE': str})
    #Chromosomal microarray set (individuals who received a genetic test)
    cma_df = pd.read_csv(sys.argv[4])
    #Location (string) where the frequent visitors case probability predictions should be written to
    fv_out = sys.argv[5]
    #Location (string) where the parameters and cross validation results for all tested models should be written to
    param_out = sys.argv[6]
    num_cpu = int(sys.argv[7])
    print('Loaded args')
    #Set cc_status (case control status) tag in demo_df to indicate which patients received a CMA
    demo_df['CC_STATUS']=0
    demo_df.loc[demo_df.GRID.isin(cma_df.GRID),'CC_STATUS']=1
    #Match controls 4:1 with the cma recipients
    print('Beginning to match controls')
    case_control_df = match_controls.new_cc_match(demo_df, demo_df.loc[demo_df.CC_STATUS==1,:].copy(), 4)
    print(case_control_df.head())
    cc_grids=list(case_control_df.GRID)
    print('matched controls')
    #Create long df for case control set
    phecodes.columns=['PERSON_ID','GRID', 'CODE','DATE']
    cc_phecodes = phecodes.loc[phecodes.GRID.isin(cc_grids)].copy().reset_index()
    if CENSOR:
        cc_phecodes = censor_codes(cc_phecodes, cma_df)
    cc_phecodes_group = cc_phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    long_cc_df = demo_df.loc[demo_df.GRID.isin(cc_grids)].merge(cc_phecodes_group, on="GRID", how="left")
    long_cc_df[cc_phecodes.CODE.unique()] = long_cc_df[cc_phecodes.CODE.unique()].fillna(0).astype(int)
    print('Created long cc df')
    ##Create long df for fv
    #fv_phecodes = phecodes.loc[phecodes.GRID.isin(demo_df.loc[demo_df.FV_STATUS==1,'GRID'])].copy().reset_index()
    #fv_phecodes_group = fv_phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    #long_fv_df = demo_df.loc[demo_df.FV_STATUS==1].merge(fv_phecodes_group, on="GRID", how="left")
    #long_fv_df[fv_phecodes.CODE.unique()] = long_fv_df[fv_phecodes.CODE.unique()].fillna(0).astype(int)
    #print('created long fv df')
    ##Get the weight sum (Phenotypic risk score)
    weights = pd.Series(weight_df.WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    print('loaded weights')
    #drop cogenital anomalies (would be cheating to use them)
    for code in ['758', '758.1', '759', '759.1']:
        if code in long_cc_df:
            long_cc_df[code]=0
        #if code in long_fv_df:
        #    long_fv_df[code]=0
    phe_list = list(weight_df.PHECODE.unique())
    anc_child_dict = create_weight_df.create_ancestry(phe_list)
    leaf_select_cc_df = create_weight_df.leaf_select_codes(long_cc_df, anc_child_dict, phe_list)
    #leaf_select_fv_df = create_weight_df.leaf_select_codes(long_fv_df, anc_child_dict, phe_list)
    summed_cc_weight = create_weight_df.get_sums(leaf_select_cc_df, weights)['weight_sum']
    #summed_fv_df = create_weight_df.get_sums(leaf_select_fv_df, weights)
    #copy over weight sum to long df, since we want all phecodes present for the ml process
    long_cc_df['weight_sum']=summed_cc_weight
    #long_fv_df['weight_sum']=summed_fv_df['weight_sum']
    print('summed weights; performed ancestry analysis w/codes')
    #Run sklearn pipeline for grid search
    phe_list = [x for x in list(weight_df.PHECODE.unique()) if x in long_cc_df.columns]
    #phe_list = [x for x in phecodes.CODE if x in long_cc_df.columns]
    phe_list.append('weight_sum')
    #Garbage collect time?
    gc.collect()
    results_df, best_estimator = cross_validation_pipeline_probabilistic.sklearn_pipeline(long_cc_df[phe_list], long_cc_df['CC_STATUS'].astype(int), num_cpu)
    results_df.to_csv(param_out,index=False)
    print('pipeline complete')
    ###retrain best parameters on entire cc set (new problem, same model)
    #best_estimator.fit(long_cc_df[phe_list], long_cc_df['CC_STATUS'].astype(int))
    #print('estimator retrained')
    ##predict on frequent visitors set (minus the cases and controls)
    #fv_df=long_fv_df.loc[~long_fv_df.GRID.isin(cc_grids)].copy()
    #fv_df = fv_df.sample(frac=1)
    #probs=best_estimator.predict_proba(fv_df[phe_list])
    #print('prediction complete')
    ##fv_df['control_prob'] = probs[:, 0]
    #fv_df['case_prob'] = probs[:, 1]
    #fv_df[['GRID','case_prob']].to_csv(fv_out,index=False)
    #print('wrote results')

if __name__=='__main__':
    main()
