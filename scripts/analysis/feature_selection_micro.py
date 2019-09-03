import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

def biclass(s):
    if s != "Normal":
        return "Abnormal"
    else:
        return s

def chi_select():
    #Read in phecodes for all individuals ###, perform lasso to find relevant features, and test their correlation with normality status
    full_phe = pd.read_csv(sys.argv[1])
    full_phe['Binary_Result'] = full_phe['Result'].apply(biclass)
    full_phe['Binary_Result'] = np.where(full_phe['Binary_Result']=='Normal', 0, 1)
    #feature selection
    phe_cols = full_phe.drop(['Binary_Result','NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','dob'], axis=1)
    X, y = phe_cols.fillna(0), full_phe['Binary_Result']
    print(X.shape)
    sel_k_clf = SelectKBest(chi2, k=5).fit(X,y)
    idxs = sel_k_clf.get_support(indices=True)
    X_new = X.iloc[:,idxs]
    #X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print(X_new.shape)
    print(X_new.head())
    #print(chi2(X,y))
    print(chi2(X_new,y))

def pca():
    df = pd.read_csv(sys.argv[1])
    phe_cols = df.drop(['NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','unclear','dob','age','years'], axis=1)
    print(phe_cols['637.0'])
    pca = PCA(n_components=2)
    #data_scaled = pd.DataFrame(preprocessing.scale(phe_cols),columns = phe_cols.columns) 
    pca.fit_transform(phe_cols)
    print(pca.explained_variance_ratio_)
    expl = pd.DataFrame(pca.components_,columns=phe_cols.columns,index = ['PC-1','PC-2'])
    print(expl)
    print(abs(expl).max(axis=1))
    s = expl.iloc[0]
    print(s[abs(s)>0.2])

def tsvd():
    df = pd.read_csv(sys.argv[1])
    phe_cols = df.drop(['NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','unclear','dob','age','years'], axis=1)
    svd = TruncatedSVD()
    svd.fit(phe_cols)
    print(svd.explained_variance_ratio_)
    print(svd.explained_variance_ratio_.sum())

def lasso():
    df = pd.read_csv(sys.argv[1])
    #phe_cols = df.drop(['NOTE_ID','GRID','ENTRY_DATE','Result','size','location','result_text','interp','indication','Method','BP_start','BP_end','ISCN','unclear','dob','age','years'], axis=1) 

if __name__=='__main__':
    #chi_select()
    pca()
    #tsvd()
