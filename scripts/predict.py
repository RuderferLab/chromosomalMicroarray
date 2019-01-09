import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_sums(cc_df, weights, unique_phecodes):
    cc_df['weight_sum'] = 0
    for phc in weights.keys():
        #Create weight sum column, where for each phecode column, if the count is nonzero the corresponding weight is added to the sum, otherwise it is ignored.
        if phc in cc_df.columns:
            cc_df['weight_sum'] += np.where(cc_df[phc]>0, weights[phc], 0)
        else:
            pass
            #print(phc)
    #print(cc_df.head())
    #print(len(cc_df['weight_sum'].unique()))
    cc_df=cc_df.merge(unique_phecodes, on='GRID', how='left').fillna(0)
    cc_df['unique_phecode_adjusted_weight']=cc_df['weight_sum']/cc_df['UNIQUE_PHECODES']
    return cc_df

'''
Same as get_sums(), but the phecode_desc is a dict mapping from phecode to the phewas_string for it (all lower case)
'''
def get_sums_no_ca(cc_df, weights, unique_phecodes, phecode_desc):
    cc_df['weight_sum_no_ca'] = 0
    excnt=0
    for phc in weights.keys():
        #Create weight sum column, where for each phecode column, if the count is nonzero the corresponding weight is added to the sum, otherwise it is ignored.
        if phc in cc_df.columns and phc in phecode_desc and "congenital" not in phecode_desc[phc]:
                cc_df['weight_sum_no_ca'] += np.where(cc_df[phc]>0, weights[phc], 0)
        elif phc in phecode_desc and "congenital" in phecode_desc[phc]:
            excnt+=1
        else:
            pass
            #print(phc)
    #print(cc_df.head())
    #print(len(cc_df['weight_sum'].unique()))
    print(excnt)
    print("Excluded count!^")
    cc_df=cc_df.merge(unique_phecodes, on='GRID', how='left').fillna(0)
    cc_df['unique_phecode_adjusted_weight']=cc_df['weight_sum_no_ca']/cc_df['UNIQUE_PHECODES']
    print(min(cc_df.UNIQUE_PHECODES.unique()))
    print(cc_df.unique_phecode_adjusted_weight.unique())
    return cc_df

def logistic_predict(df, ca):
    #Get train and test
    #print(df.head())
    if ca:
        train_x, test_x, train_y, test_y = train_test_split(df[['weight_sum', 'UNIQUE_PHECODES']], df['cc_status'], test_size=.25)
        #train_x = train_x.reshape(-1,1)
        #test_x=test_x.reshape(-1,1)
    else:
        train_x, test_x, train_y, test_y = train_test_split(df['weight_sum_no_ca'], df['cc_status'], test_size=.25)
    #set up cross validation strategy with logistic regression
    #print(train_x.reshape(-1,1))
    clf=LogisticRegression()#(class_weight='balanced')
    clf.fit(train_x, train_y)
    print(clf.score(test_x, test_y))
    y_pred = clf.predict(test_x)
    conf=confusion_matrix(test_y, y_pred)
    print(conf)
    tn, fp, fn, tp = conf.ravel()
    print("TN:"+str(tn))
    print("FP:"+str(fp))
    print("FN:"+str(fn))
    print("TP:"+str(tp))
    #Get ROC Curve
    fpr, tpr, thresholds = roc_curve(test_y, y_pred)
    roc_auc = auc(fpr, tpr)

    #Plot the curve and save it
    plt.title('Receiver Operating Characteristic: Weighted Sum')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC_weighted_sum_unique_codes.png")

    #cv_results=cross_validate(logreg, train_x, train_y, cv=3, return_estimator=True)
    #print(cv_results['test_score'])
    #parameters = {}
    #ss = ShuffleSplit(n_splits=3, test_size=0.25)
    #for train_ind, val_ind in ss.split(train_x):


def chitests(ldf, weights):
    cases = []
    controls = []
    results = dict()
    for code in weights.keys():
        if code in ldf:
            cont = ldf[code].loc[ldf.cc_status==0].sum()
            case = ldf[code].loc[ldf.cc_status==1].sum()
            cases.append(case)
            controls.append(cont)
            chi, p = chisquare([cont,case])
            results[code]=(chi, p)
        else:
            pass
            #print("NOT IN")
            #print(code)
    #print(np.array([controls,cases]).T)
    #print(chisquare(np.array([controls,cases]).T))
    #print(results.items())
    return results

def save_res_table(res_dict, out):
    pd.DataFrame.from_dict(res_dict, orient='index').to_csv(out)


#python predict.py long_cc weights unique_phecodes phecode_translate(for desc)
if __name__=="__main__":
    #get phecode_desc from phecode table
    ph = pd.read_table(sys.argv[4])
    ph['phewas_string']=ph['category_string'].str.lower()
    ph['phewas_code']=ph['phewas_code'].astype(str)
    pdesc=pd.Series(ph.category_string.values,index=ph.phewas_code).to_dict()

    weight_df=pd.read_csv(sys.argv[2])
    weights=pd.Series(weight_df.LOG_WEIGHT.values,index=weight_df.PHECODE.astype(str)).to_dict()
    df = get_sums(pd.read_csv(sys.argv[1]), weights, pd.read_csv(sys.argv[3]))[['weight_sum', 'UNIQUE_PHECODES', 'cc_status']]
    logistic_predict(df, True)

    #df_no_ca = get_sums_no_ca(pd.read_csv(sys.argv[1]), weights, pd.read_csv(sys.argv[3]), pdesc)[['weight_sum_no_ca', 'cc_status']]
    #print("No CA!")
    #logistic_predict(df_no_ca, False)
