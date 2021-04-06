import pandas as pd
import numpy as np
import sys

#includes itself as a child
def is_child_present(grid, phecode, indiv_pres, key, anc_ch):
    children = list(anc_ch.loc[anc_ch.phecode_ancestor==phecode, 'phecode_child'])
    for c in children:
        if indiv_pres[grid][key[c]]:
            return True
    return False

def rollup(phe_path, child_anc_path):
    phecodes = pd.read_csv(phe_path, dtype=str)
    child_anc = pd.read_csv(child_anc_path, dtype=str)

    prev_dict = dict()
    phe_key = dict()
    individual_presence = dict()
    #each individual has a list of counts, corresponding in order to the unique_codes list
    #increase counts according to whether a child is present
    unique_codes = list(phecodes.PHECODE.unique())
    for i in range(len(unique_codes)):
        phe_key[unique_codes[i]]=i
    for indv in phecodes.GRID.unique():
        #intialize len(unique_codes) list containing false for all GRIDs
        individual_presence[indv] = [False]*len(unique_codes)
    for i in range(phecodes.shape[0]):
        #check if child is present for given phecode and individual
        #if it is not, set it to True for that individual and increment the count in the prev_dict
        #also increment all the parents of this code in the prev dict
        if not is_child_present(phecodes.iloc[i].GRID, phecodes.iloc[i].PHECODE, individual_presence, phe_key, child_anc):
            individual_presence[phecodes.iloc[i].GRID][phe_key[phecodes.iloc[i].PHECODE]]=True
            #for all parents of the phecode, increase count by 1
            parents = list(child_anc.loc[child_anc.phecode_child==phecodes.iloc[i].PHECODE, 'phecode_ancestor'])
            for code in parents:
                if code in prev_dict:
                    prev_dict[code]+=1
                else:
                    prev_dict[code] = 1
        else:
            #It is, so this individual has already been counted for this code
            #This is because any time a child is found, each of it's parents are incremented
            #Therefore this individual has already been counted in the prevalence for this code, since a child of this code has been counted
            individual_presence[phecodes.iloc[i].GRID][phe_key[phecodes.iloc[i].PHECODE]] = True
    return prev_dict, phecodes.GRID.unique().shape[0]

def rollup_file(phe_path, child_anc_path):
    phe_file = open(phe_path, 'r')
    phedf = pd.read_Csv(phe_path, dtype=str)
    child_anc = pd.read_csv(child_anc_path, dtype=str)

    prev_dict = dict()
    phe_key = dict()
    individual_presence = dict()
    unique_codes = list(phedf.PHECODE.unique())
    print('initialized dicts')
    for i in range(len(unique_codes)):
        phe_key[unique_codes[i]]=i
    print('initialized key')
    for indv in phedf.GRID.unique():
        individual_presence[indv] = [False]*len(unique_codes)
    print('initialized indv presence')
    for line in phe_file:
        #split line 
        grid, phecode, _ = line.strip('\n').split(',')
        if not is_child_present(grid, phecode, individual_presence, phe_key, child_anc):
            individual_presence[grid][phe_key[phecode]]=True
            parents = list(child_anc.loc[child_anc.phecode_child==phecode, 'phecode_ancestor'])
            for code in parents:
                if code in prev_dict:
                    prev_dict[code] += 1
                else:
                    prev_dict[code] = 1
        else:
            individual_presence[grid][phe_key[phecode]] = True
    print('returning prevalence results')
    return prev_dict, phedf.GRID.unique().shape[0]

def create_weights(prev_dict, total_pop):
    prev = pd.DataFrame()
    prev['codes'] = prev_dict.keys()
    prev['counts'] = prev_dict.values()
    prev['inv_prevalence'] = total_pop/prev['counts']
    #np.log is ln
    prev['ln'] = np.log(prev['inv_prevalence'])
    return prev

def run_rollup_file():
    prev_dict, total_pop = rollup_file(sys.argv[1], sys.argv[2])
    prev_df = create_weights(prev_dict, total_pop)
    prev_df.to_csv(sys.argv[3])

'''
args:
sys.argv[1] = path to phecodes (frequent visitors)
sys.argv[2] = path to ancestor to child key for phecodes
sys.argv[3] = path to output
'''
if __name__=='__main__':
    prev_dict, total_pop = rollup(sys.argv[1], sys.argv[2])
    prev_df = create_weights(prev_dict, total_pop)
    prev_df.to_csv(sys.argv[3])
