import pandas as pd
import sys


if __name__=='__main__':
    df = pd.read_csv(sys.argv[1])
    df.columns = ['GRID', 'CODE', 'COUNT']
    wide_df = df.groupby(['GRID', 'CODE']).sum().unstack(fill_value=0).astype(int).reset_index()
    wide_df.to_csv(sys.argv[2], index=False)
