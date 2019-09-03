import sys
import numpy as np
import pandas as pd

'''
Input: 
    df: Full dataframe with phecodes, identifiers, and covariates.
    phecode_list: list of unique phecodes
Output: Only the columns corresponding to the phecodes
'''
def phecodes_only(df, phecode_list):
    present = [p for p in phecode_list if p in df]
    return df[present].astype(int)


'''
Input: Dataframe containing only columns corresponding to phecodes
Output: Same dataframe, but with cells whose values>0 replaced with 1
'''
def binary_transform(df):
    #np.where(cc_df[phc]>0, weights[phc], 0)
    #df = np.where(df>0, 1, 0)
    df[df>0] = 1
    return df

'''
Input:
    df: Binary transformed matrix of phecodes
Output: Tuple containing the correlation matrix, as well as a copy containing only the upper right half of the matrix (rest NA), and a matrix with the correlations ranked pairwise
'''
def correlation_matrix(df):
    correlation = df.corr()
    #Get only one half
    half = correlation.where(np.triu(np.ones(correlation.shape)).astype(np.bool))
    ranked = half.stack().reset_index()
    ranked.columns = ['phecode_1','phecode_2','correlation']
    ranked = ranked.sort_values(by='correlation', ascending=False)
    ranked = ranked.loc[ranked.phecode_1 != ranked.phecode_2]
    ranked['distinct_grids'] = ranked.apply(grid_count, args=(df,), axis=1)
    return (correlation, half, ranked)

'''
Input:
    s: row from ranked_mat: dataframe with phecode pairs and their correlation
    full_df: The full long form dataframe
Output: ranked_mat, but with an additional column "distinct_grids", which counts the number of cases which have both phecodes in that row
'''
def grid_count(s, full_df):
    p1 = s['phecode_1']
    p2 = s['phecode_2']
    return full_df.loc[(full_df[p1]>0) & (full_df[p2]>0)].shape[0]

#python program df weights outputname
if __name__=='__main__':
    df = pd.read_csv(sys.argv[1], dtype=str)
    weights = pd.read_csv(sys.argv[2], dtype=str)
    out = sys.argv[3]
    phe_list = weights.PHECODE.unique()
    
    #Begin process
    phe_df = phecodes_only(df, phe_list)
    phe_df = binary_transform(phe_df)
    corr_df, half_df, ranked_corrs = correlation_matrix(phe_df)
    print(ranked_corrs.head())

    #Write output
    corr_df.to_csv(out+'_corr_df_full.csv')
    half_df.to_csv(out+'_corr_df_half.csv')
    ranked_corrs.to_csv(out+'_ranked_corrs.csv', index=False)
