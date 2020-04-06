import pandas as pd
import sys

'''
in: codes
    a list containing all unique codes present
'''
def create_ancestry(codes):
    df = pd.DataFrame()
    anc_key = dict()
    for anc_code in codes:
        for child_code in codes:
            if len(child_code)<len(anc_code):
                pass
            elif anc_code==child_code[:len(anc_code)]:
                if anc_code in anc_key:
                    anc_key[anc_code].append(child_code)
                else:
                    anc_key[anc_code]=[child_code]
            else:
                pass
    #Now have a key matching each ancestor code to the list of all of the child codes belonging to it
    anc_codes = []
    child_codes = []
    for k, v in anc_key:
        for code in v:
            anc_codes.append(k)
            child_codes.append(code)
    df['phecode_ancestor']= anc_codes
    df['phecode_child']= child_codes
    return df

if __name__=='__main__':
    #load in dataframe of all phecodes present -- phecodes column labelled PHECODE
    df = pd.read_csv(sys.argv[1], dtype=str)
    ancestry_df = create_ancestry(list(df.PHECODE.unique()))
    ancestry_df.to_csv(sys.argv[2], index=False)
