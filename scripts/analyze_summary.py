import sys
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

def biclass(s):
    if s != "Normal":
        return "Abnormal"
    else:
        return s

def combine_string(s):
    if 'delay' in s.lower() or 'developmental' in s.lower() or 'lack of expected normal' in s.lower():
        return 'developmental delay'
    elif 'congenital' in s.lower():
        return 'congenital anomalies'
    else:
        return s


def tri_cats():
    #load summary
    df = pd.read_csv(sys.argv[1])
    #compare three classes (abnormal, normal, and unclear), on number of unique phecodes
    df['Binary-Result'] = df['result'].apply(biclass)
    #Change unclear results to special category
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result']) 
    sns.swarmplot(x="Tri-Result", y="unique_categories", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def tri_phe_un_pt():
    #load summary
    df = pd.read_csv(sys.argv[1])
    #compare three classes (abnormal, normal, and unclear), on number of unique phecodes
    df['Binary-Result'] = df['result'].apply(biclass)
    #Change unclear results to special category
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result'])
    fig, ax = plt.subplots(figsize=(9,9))
    sns.swarmplot(x="Tri-Result", y="pretest_phecodes_unique", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()


def region_age():
    df = pd.read_csv(sys.argv[1])
    locs = pd.DataFrame(df.groupby("location").size()).reset_index()
    locs = locs.loc[locs[0]>3]
    locs = locs.loc[locs.location!='-']
    fig, ax = plt.subplots(figsize=(28,7))
    fil = df.merge(locs,on="location", how='inner', )
    sns.swarmplot(x="location", y="age", data=fil)
    plt.savefig(sys.argv[2])
    plt.clf()


def ty_sum():
    df = pd.read_csv(sys.argv[1])
    df['record_len'] = df['record_len']/365
    fig, ax = plt.subplots(figsize=(21,8))
    sns.regplot(x="record_len",y="phecode_sum",data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def ty_sum_pt():
    df = pd.read_csv(sys.argv[1])
    df['record_len'] = df['record_len']/365
    fig, ax = plt.subplots(figsize=(21,8))
    sns.regplot(x="record_len",y="pretest_sum",data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def ty_uniq_pt():
    df = pd.read_csv(sys.argv[1])
    df['record_len'] = df['record_len']/365
    fig, ax = plt.subplots(figsize=(21,8))
    sns.regplot(x="record_len",y="pretest_phecodes_unique",data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def age_res():
    df = pd.read_csv(sys.argv[1])
    df['Binary-Result'] = df['result'].apply(biclass)
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result'])
    fig, ax = plt.subplots(figsize=(9,9))
    sns.swarmplot(x="Tri-Result", y="age", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def ty_res():
    df = pd.read_csv(sys.argv[1])
    df['record_len'] = df['record_len']/365
    df['Binary-Result'] = df['result'].apply(biclass)
    df['Tri-Result'] = np.where(df['unclear']==1, 'Unclear', df['Binary-Result'])
    fig, ax = plt.subplots(figsize=(9,9))
    sns.boxplot(x="Tri-Result", y="record_len", data=df)
    plt.savefig(sys.argv[2])
    plt.clf()

def age_ind():
    df = pd.read_csv(sys.argv[1])
    locs = pd.DataFrame(df.groupby("indication").size()).reset_index()
    locs = locs.loc[locs[0]>10]
    fig, ax = plt.subplots(figsize=(28,7))
    fil = df.merge(locs,on="indication", how='inner')
    fil['indication'] = fil.indication.apply(combine_string)
    sns.swarmplot(x="indication", y="age", data=fil)
    plt.savefig(sys.argv[2])
    plt.clf()

def predict():
    df = pd.read_csv(sys.argv[1])
    df['Binary-Result'] = df['result'].apply(biclass)
    df['Binary-Result'] = np.where(df['Binary-Result']=='Normal', 0, 1)
    print(df.columns.values)
    X, y = df[['pretest_sum', 'pretest_phecodes_unique']], df['Binary-Result']
    #df[['phecode_sum','pretest_sum','pretest_phecodes_unique','phecodes_unique','unique_categories']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print("f1")
    print(f1_score(y_test, clf.predict(X_test)))
    print(confusion_matrix(y_test, clf.predict(X_test)))
    ##
    #knn
    #print('knn')
    #for k in [2,3,4,5,6,7,8]:
    #    print(k)
    #    kn_clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    #    print(kn_clf.score(X_test, y_test))
    #    print("f1")
    #    print(f1_score(y_test, kn_clf.predict(X_test)))
    #    print(confusion_matrix(y_test, kn_clf.predict(X_test)))
    #rf
    print('rf')
    rf = RandomForestClassifier().fit(X_train, y_train)
    pred = rf.predict(X_test)
    print(rf.score(X_test, y_test))
    print("f1")
    print(f1_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print("NB")
    nb = GaussianNB().fit(X_train, y_train)
    print(nb.score(X_test, y_test))
    nb_pred = nb.predict(X_test)
    print("f1")
    print(f1_score(y_test, nb_pred))
    print(confusion_matrix(y_test, nb_pred))


if __name__ =="__main__":
    #region_age()
    #ty_sum()
    #ty_sum_pt()
    #ty_uniq_pt()
    #tri_cats()
    #tri_phe_un_pt()
    #age_res()
    #ty_res()
    #age_ind()
    predict()
