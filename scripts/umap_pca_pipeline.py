import numpy as np
import sys
import umap
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import chi2_contingency as chi2_c

#Plan: Reduce full feature set using PCA to size of 50, UMAP to 10

def create_pc_embedding(df, embed_cols, num_pc):
    #PCA embedding
    pca = PCA(n_components=num_pc)
    pc_embedding = pca.fit_transform(df[embed_cols])
    return (pc_embedding, pca)

def create_umap_embedding(pc_df, num_ump, met):
    ump = umap.UMAP(metric=met, n_components=num_ump)
    ump_embedding = ump.fit_transform(pc_df)
    return ump_embedding

def prediction_pipeline(predictors, df, target):
    #Pick train test split
    X_train, X_test, y_train, y_test = train_test_split(df, target.astype(int), test_size=0.25)
    #Test provided predictors
    for p in predictors:
        p.fit(X_train, y_train)
        print(p)
        print(confusion_matrix(y_test, p.predict(X_test)))
        print(p.score(X_test, y_test))


if __name__=='__main__':
    df = pd.read_csv(sys.argv[1], dtype={'location':str, 'Result':str})
    #Drop rows which have no demographic info
    df=df.drop(df[df.BIRTH_DATETIME=='0'].index)
    phecodes = pd.read_csv(sys.argv[2], dtype=str)
    #Binarize and get columns for phecodes
    phe_list = [phe for phe in list(phecodes.PHECODE.unique()) if phe in df]
    phedf = df.loc[:, phe_list]
    phedf[phedf>0] = 1
    df[phe_list] = phedf
    #Run process for dimensionality reduction
    pc_emb,_ = create_pc_embedding(df, phe_list, 50)
    pc_emb = pd.DataFrame(pc_emb)
    ump_emb = create_umap_embedding(pc_emb, 10, 'euclidean')
    df_reduced = pd.DataFrame(ump_emb, columns=['UMP-'+str(i+1) for i in range(10)])
    #Add relevant covariate columns if they will be used for prediction?
    #Run prediction
    prediction_pipeline([GaussianNB(), LogisticRegression()], df_reduced, df['CC_STATUS'])
