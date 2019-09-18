import dask.dataframe as dd
import pandas as pd
import sys


if __name__=='__main__':
    df_icd = dd.read_csv(sys.argv[1], dtype=str, sep='|')
    grouped_icd = pd.DataFrame(df_icd.groupby(['GRID','CODE']).size().compute())
    grouped_icd.to_csv(sys.argv[2])
