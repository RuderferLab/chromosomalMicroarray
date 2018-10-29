import pandas as pd
import sys

def translate(s, tra):
    if len(tra.loc[tra.phewas_code==s].category.unique())>0:
        return tra.loc[tra.phewas_code==s].category.unique()[0]
    else:
        print(s)
        return -1

if __name__=='__main__':
    translation = pd.read_table(sys.argv[1])
    phecodes = pd.read_csv(sys.argv[2])
    phecodes['cat'] = phecodes['code'].apply(translate, args=(translation,))
    phecodes.to_csv(sys.argv[3])
