import sys
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

#python master.py demographic_data phecodes weights CMA fv_out param_out
def main():
    #Load in all inputs
    demo_df = pd.read_csv(sys.argv[1])
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    weight_df = pd.read_csv(sys.argv[3], dtype={'PHECODE': str})
    cma_df = pd.read_csv(sys.argv[4])
    fv_out = sys.argv[5]
    param_out = sys.argv[6]
    #Set cc_status tag in demo_df
    demo_df['CC_STATUS']=0
    demo_df.loc[demo_df.GRID.isin(cma_df.GRID),'CC_STATUS']=1
    #Match controls #fullpop cases
    case_control_df = match_controls.cc_match(demo_df, demo_df.loc[demo_df.cc_status==1,:])
    cc_grids=list(case_control_df.GRID)
    #Create long df for everyone
    phecodes.columns=['GRID', 'CODE']
    phecodes_group = phecodes.groupby(['GRID','CODE']).size().unstack().fillna(0).astype(int).reset_index()
    long_df = demo_df.merge(phecodes_group, on="GRID", how="outer")
    long_df[phecodes.CODE.unique()] = df[phecodes.CODE.unique()].fillna(0).astype(int)
    #Get the weight sum
    weights = pd.Series(weight_df.WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    #drop cog anoms
    for code in ['758', '758.1', '759', '759.1']:
        if code in long_df:
            long_df[code]=0
    phe_list = list(weight_df.PHECODE.unique())
    anc_child_dict = create_weight_df.create_ancestry(phe_list)
    long_df = create_weight_df.leaf_select_codes(long_df, anc_child_dict, phe_list)
    summed_df = create_weight_df.get_sums(long_df, weights)
    #Run sklearn pipeline for grid search
    phe_list = [x for x in phecodes.CODE if x in summed_df.columns]
    results_df, best_estimator = cross_validation_pipeline_probabilistic.sklearn_pipeline(summed_df[phe_list+['weight_sum']], summed_df['CC_STATUS'].astype(int))
    results_df.to_csv(param_out,index=False)
    #retrain best parameters on entire cc set
    best_estimator.fit(summed_df.loc[summed_df.GRID.isin(cc_grids),phe_list+['weight_sum']], summed_df.loc[summed_df.GRID.isin(cc_grids),'CC_STATUS'].astype(int))
    #predict on frequent visitors set
    fv_df=summed_df.loc[summed_df['fv_status']==1].copy()
    fv_df = fv_df.sample(frac=1)
    probs=best_estimator.predict_proba(fv_df[phe_list+['weight_sum']])
    #fv_df['control_prob'] = probs[:, 0]
    fv_df['case_prob'] = probs[:, 1]
    fv_df[['GRID','case_prob']].to_csv(fv_out,index=False)

if __name__=='__main__':
    main()
