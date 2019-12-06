import dask.dataframe as dd
import pandas as pd
import sys


if __name__=='__main__':
    df_icd = dd.read_csv(sys.argv[1], dtype=str, sep='|')
    icd_type = sys.argv[2]
    grouped_icd = pd.DataFrame(df_icd.loc[df_icd.VOCABULARY_ID==icd_type].groupby(['GRID','CODE']).size().compute())
    grouped_icd.to_csv(sys.argv[3])
