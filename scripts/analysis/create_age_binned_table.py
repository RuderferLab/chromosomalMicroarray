#goal: create a table with age-bin, genet_status pos count, gclin pos count, as well as auroc/ap for gclin, genet_status, and the combo of the two. Additionally select threshold for each and give appropriate standard stats
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

'''
Gets stats at a certain selection threshold (given as a count of the top n individuals selected) for various phenotypes
'''
def count_bin_stats(age_bin, df, n):
    #select age bin
    within_bin_df = df.loc[(df['AGE_YEARS']>=age_bin[0]) & (df['AGE_YEARS']<=age_bin[1])].copy(deep=True)
    within_bin_df['selected'] = 0
    within_bin_nlargest = within_bin_df.nlargest(n, columns='case_prob')
    grids = within_bin_nlargest['GRID']
    within_bin_df.loc[within_bin_df['GRID'].isin(grids), 'selected'] = 1
    #counts
    total_count = within_bin_df.shape[0]
    genet_status_count = within_bin_df.loc[within_bin_df['genet_status']==1].shape[0]
    gclin_count = within_bin_df.loc[within_bin_df['gClin']==1].shape[0]
    expanded_count = within_bin_df.loc[within_bin_df['expanded_genet']==1].shape[0]
    biovu_count = within_bin_df.loc[within_bin_df['in_biovu']==1].shape[0]
    biovu_selected_count =  within_bin_df.loc[(within_bin_df['in_biovu']==1)&(within_bin_df['selected']==1)].shape[0]
    gene_cpt_count = within_bin_df.loc[within_bin_df['GENE_CPT']==1].shape[0]
    #auroc/ap/other stats for the above
    aurocs = []
    aps = []
    #following stats are calculated under the selection criteria of the top n candidates via case_prob as positives
    tps = []
    fps = []
    tns = []
    fns = []
    for var in ['gClin', 'genet_status', 'expanded_genet', 'GENE_CPT']:
        aurocs.append(roc_auc_score(within_bin_df[var], within_bin_df['case_prob']))
        aps.append(average_precision_score(within_bin_df[var], within_bin_df['case_prob']))
        #confusion matrix
        conf = confusion_matrix(within_bin_df[var], within_bin_df['selected'])
        tn, fp, fn, tp = conf.ravel()
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
    #do combined pheno 2 first
    comb_within_bin_df = within_bin_df.dropna(subset=['combined_pheno_exp']).copy(deep=True)
    aurocs.append(roc_auc_score(comb_within_bin_df['combined_pheno_exp'], comb_within_bin_df['case_prob']))
    aps.append(average_precision_score(comb_within_bin_df['combined_pheno_exp'], comb_within_bin_df['case_prob']))
    comb_within_bin_nlargest = comb_within_bin_df.nlargest(n, columns='case_prob')
    comb_within_bin_df['selected'] = 0
    grids = comb_within_bin_nlargest['GRID']
    comb_within_bin_df.loc[comb_within_bin_df['GRID'].isin(grids), 'selected'] = 1
    conf = confusion_matrix(comb_within_bin_df['combined_pheno_exp'], comb_within_bin_df['selected'])
    tn, fp, fn, tp = conf.ravel()
    tps.append(tp)
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)
    #do combined pheno seperately because we need to drop elements that are na
    within_bin_df = within_bin_df.dropna(subset=['combined_pheno'])
    aurocs.append(roc_auc_score(within_bin_df['combined_pheno'], within_bin_df['case_prob']))
    aps.append(average_precision_score(within_bin_df['combined_pheno'], within_bin_df['case_prob']))
    within_bin_nlargest = within_bin_df.nlargest(n, columns='case_prob')
    within_bin_df['selected'] = 0
    grids = within_bin_nlargest['GRID']
    within_bin_df.loc[within_bin_df['GRID'].isin(grids), 'selected'] = 1
    #within_bin_nlargest['selected'] = 1
    conf = confusion_matrix(within_bin_df['combined_pheno'], within_bin_df['selected'])
    tn, fp, fn, tp = conf.ravel()
    tps.append(tp)
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)
    #return
    return total_count, genet_status_count, gclin_count, expanded_count, aurocs, aps, tps, fps, tns, fns, biovu_count, biovu_selected_count, gene_cpt_count

'''
Input:
    df: csv containing case_prob, control_prob, AGE_YEARS, GRID, GENE_CPT, in_biovu, gClin, genet_status, and expanded_genet
    n: number to select
    out: file to write table to
'''
if __name__=='__main__':
    big_table = pd.DataFrame()
    big_table['age_bin']=[(32,54),(16,38),(20,41),(50,73),(7,28),(0,16),(62,86),(50,72),(0,20),(41,63),(18,34),(48,69),(76,98)]
    #read in merged fv results
    df = pd.read_csv(sys.argv[1])
    #select n largest to be labeled
    n = int(sys.argv[2])
    #create new phenotype if not present -- combine gclin and genet into combined_pheno
    if 'combined_pheno' not in df:
        df['combined_pheno'] = np.nan
        df.loc[df['genet_status']==0, 'combined_pheno'] = 0
        df.loc[df['gClin']==1, 'combined_pheno'] = 1
    if 'combined_pheno_exp' not in df:
        df['combined_pheno_exp'] = np.nan
        df.loc[df['expanded_genet']==0, 'combined_pheno_exp'] = 0
        df.loc[df['gClin']==1, 'combined_pheno_exp'] = 1
    #for each age bin, find the count of individuals who belond in it
    total_counts = []
    genet_counts = []
    gclin_counts = []
    expanded_genet_counts = []
    biovu_counts = []
    biovu_selected_counts = []
    cpt_counts = []
    #
    genet_aurocs = []
    gclin_aurocs = []
    combined_aurocs = []
    expanded_genet_aurocs = []
    combined2_aurocs = []
    cpt_aurocs = []
    #
    genet_aps = []
    gclin_aps = []
    combined_aps = []
    expanded_genet_aps = []
    combined2_aps = []
    cpt_aps = []
    #
    genet_tps = []
    gclin_tps = []
    combined_tps = []
    expanded_genet_tps = []
    combined2_tps = []
    cpt_tps = []
    #
    genet_fps = []
    gclin_fps = []
    combined_fps = []
    expanded_genet_fps = []
    combined2_fps = []
    cpt_fps = []
    #
    genet_tns = []
    gclin_tns = []
    combined_tns = []
    expanded_genet_tns = []
    combined2_tns = []
    cpt_tns = []
    #
    genet_fns = []
    gclin_fns = []
    combined_fns = []
    expanded_genet_fns = []
    combined2_fns = []
    cpt_fns = []
    #
    for ab in big_table['age_bin']:
        total_count, genet_status_count, gclin_count, expanded_genet_count, aurocs, aps, tps, fps, tns, fns, bv_count, bv_selected_count, cpt_count = count_bin_stats(ab, df, n)
        total_counts.append(total_count)
        genet_counts.append(genet_status_count)
        gclin_counts.append(gclin_count)
        expanded_genet_counts.append(expanded_genet_count)
        biovu_counts.append(bv_count)
        biovu_selected_counts.append(bv_selected_count)
        cpt_counts.append(cpt_count)
        #
        genet_aurocs.append(aurocs[1])
        gclin_aurocs.append(aurocs[0])
        combined_aurocs.append(aurocs[5])
        expanded_genet_aurocs.append(aurocs[2])
        combined2_aurocs.append(aurocs[4])
        cpt_aurocs.append(aurocs[3])
        #
        genet_aps.append(aps[1])
        gclin_aps.append(aps[0])
        combined_aps.append(aps[5])
        expanded_genet_aps.append(aps[2])
        combined2_aps.append(aps[4])
        cpt_aps.append(aps[3])
        #
        genet_tps.append(tps[1])
        gclin_tps.append(tps[0])
        combined_tps.append(tps[5])
        expanded_genet_tps.append(tps[2])
        combined2_tps.append(tps[4])
        cpt_tps.append(tps[3])
        #
        genet_fps.append(fps[1])
        gclin_fps.append(fps[0])
        combined_fps.append(fps[5])
        expanded_genet_fps.append(fps[2])
        combined2_fps.append(fps[4])
        cpt_fps.append(fps[3])
        #
        genet_tns.append(tns[1])
        gclin_tns.append(tns[0])
        combined_tns.append(tns[5])
        expanded_genet_tns.append(tns[2])
        combined2_tns.append(tns[4])
        cpt_tns.append(tns[3])
        #
        genet_fns.append(fns[1])
        gclin_fns.append(fns[0])
        combined_fns.append(fns[5])
        expanded_genet_fns.append(fns[2])
        combined2_fns.append(fns[4])
        cpt_fns.append(fns[3])
    #fill out table with values
    big_table['total_count'] = total_counts
    big_table['genet_status_pos_count'] = genet_counts
    big_table['gclin_pos_count'] = gclin_counts
    big_table['expanded_genet_status_pos_count'] = expanded_genet_counts
    big_table['biovu_count'] = biovu_counts
    big_table['biovu_selected_count'] = biovu_selected_counts
    big_table['gene_cpt_count'] = cpt_count
    #
    big_table['genet_auroc'] = genet_aurocs
    big_table['gClin_auroc'] = gclin_aurocs
    big_table['combined_phenotype_auroc'] = combined_aurocs
    big_table['expanded_genet_auroc'] = expanded_genet_aurocs
    big_table['combined_exp_auroc'] = combined2_aurocs
    big_table['gene_cpt_auroc'] = cpt_aurocs
    #
    big_table['genet_ap'] = genet_aps
    big_table['gClin_ap'] = gclin_aps
    big_table['combined_phenotype_ap'] = combined_aps
    big_table['expanded_genet_ap'] = expanded_genet_aps
    big_table['combined_exp_ap'] = combined2_aps
    big_table['gene_cpt_ap'] = cpt_aps
    #
    big_table['genet_tp'] = genet_tps
    big_table['gClin_tp'] = gclin_tps
    big_table['combined_phenotype_tp'] = combined_tps
    big_table['expanded_genet_tp'] = expanded_genet_tps
    big_table['combined_exp_tp'] = combined2_tps
    big_table['gene_cpt_tp'] = cpt_tps
    #
    big_table['genet_fp'] = genet_fps
    big_table['gClin_fp'] = gclin_fps
    big_table['combined_phenotype_fp'] = combined_fps
    big_table['expanded_genet_fp'] = expanded_genet_fps
    big_table['combined_exp_fp'] = combined2_fps
    big_table['gene_cpt_fp'] = cpt_fps
    #
    big_table['genet_tn'] = genet_tns
    big_table['gClin_tn'] = gclin_tns
    big_table['combined_phenotype_tn'] = combined_tns
    big_table['expanded_genet_tn'] = expanded_genet_tns
    big_table['combined_exp_tn'] = combined2_tns
    big_table['gene_cpt_tn'] = cpt_tns
    #
    big_table['genet_fn'] = genet_fns
    big_table['gClin_fn'] = gclin_fns
    big_table['combined_phenotype_fn'] = combined_fns
    big_table['expanded_genet_fn'] = expanded_genet_fns
    big_table['combined_exp_fn'] = combined2_fns
    big_table['gene_cpt_fn'] = cpt_fns
    #write out
    big_table.to_csv(sys.argv[3], index=False)
