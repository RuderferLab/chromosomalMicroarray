import sys
from pandas import read_csv
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn

#'Gain', 'Gain || Homozygosity', 'Gain || Mosaic', 'Homozygosity', 'Loss', 'Loss || Gain', 'Loss || Gain || Mosa', 'Loss || Mosaic', 'Mosaic', 'Normal', 'abnormal X:Y'
def main():
    df = read_csv(sys.argv[1])

    print(df.columns.values)

    print(df.head())

    df['sum'] = df.drop(["total_years_at_v", "age", 'Gain', 'Gain || Homozygosity', 'Gain || Mosaic', 'Homozygosity', 'Loss', 'Loss || Gain', 'Loss || Gain || Mosa', 'Loss || Mosaic', 'Mosaic', 'Normal','GRID', 'abnormal X:Y'], axis=1).sum(axis=1)
    print(df['sum'].mean())
    print(df['sum'].head())

    df['Abnormal'] = df[['Gain', 'Gain || Homozygosity', 'Gain || Mosaic', 'Homozygosity', 'Loss', 'Loss || Gain', 'Loss || Gain || Mosa', 'Loss || Mosaic', 'Mosaic', 'abnormal X:Y']].sum(axis=1)


    print("Correlation of abnormal count with sum")
    print(pearsonr(df['Abnormal'], df['sum']))

    print("Correlation of normal count with sum")
    print(pearsonr(df['Normal'], df['sum']))

    #Make histogram of age
    seaborn.distplot(df['age'], hist=True, kde=False, rug=False)
    plt.savefig(sys.argv[2])
    plt.clf()
    seaborn.distplot(df['total_years_at_v'], hist=True, kde=False, rug=False)
    plt.savefig(sys.argv[3])
    print("Mean of age")
    print(df['age'].mean())
    print("Standard deviation of age")
    print(df['age'].std())
    print("Mean of total years")
    print(df['total_years_at_v'].mean())
    print("Standard deviation of total years")
    print(df['total_years_at_v'].std())

    print("Correlation of abnormal count with total years at vanderbilt")
    print(pearsonr(df['Abnormal'], df['total_years_at_v']))
    print("Correlation of normal count with total years at vanderbilt")
    print(pearsonr(df['Normal'], df['total_years_at_v']))
    
    print("Correlation of abnormal count with age")
    print(pearsonr(df['Abnormal'], df['age']))
    print("Correlation of normal count with age")
    print(pearsonr(df['Normal'], df['age']))

    #Get info specific to phenotype (filter by 3 or more)
    codes = [x for x in df.columns.values if x not in ["total_years_at_v", "age", 'Gain', 'Gain || Homozygosity', 'Gain || Mosaic', 'Homozygosity', 'Loss', 'Loss || Gain', 'Loss || Gain || Mosa', 'Loss || Mosaic', 'Mosaic', 'Normal', 'abnormal X:Y','GRID']]
    outfile = sys.argv[4]
    out = open(outfile, 'w')
    out.write("code,mean,std,median\n")
    for pcode in codes:
        if pcode in df and df[pcode].sum() > 3:
            sel = df.loc[df[pcode]>0]['age']
            out.write(str(pcode)+','+str(sel.mean())+','+str(sel.std())+','+str(sel.median())+'\n')
            del sel
        else:
            if pcode not in df:
                print(pcode)
            pass
    out.close()

    ids = read_csv(sys.argv[5])
    ids = ids.groupby("GRID").size()
    plt.clf()
    seaborn.distplot(ids, hist=True, kde=False, rug=False)
    plt.savefig("icd_plot.png")


    '''
    #Get info specific to region (filter by 3 or more)
    regions = df.loc[df.location!="-"].location.value_counts().reset_index()
    regions = regions.loc[regions.location>3]
    print(regions.head())
    outfile = sys.argv[5]
    out = open(outfile, 'w')
    out.write("region,count,mean,median,std\n")
    for i in range(len(regions)):
        check = str(regions.iloc[i]['index'])
        sel = df.loc[df['location']==check]['age']
        out.write(str(regions.iloc[i]['index'])+','+str(regions.iloc[i]['location'])+','+str(sel.mean())+','+str(sel.median())+','+str(sel.std())+'\n')
        del sel
    out.close()
    '''

if __name__=="__main__":
    main()
