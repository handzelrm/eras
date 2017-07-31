import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, tree, svm
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from itertools import product
import time
import multiprocessing

#used to add parent directory and reload module
import os
parent_dir = os.path.join(os.getcwd(),os.pardir)
import sys
sys.path.append(parent_dir)
import project_modules
import importlib
importlib.reload(project_modules)

def split_eras(raw_pickle,processed_pickle,eras_dt):
    """
    Will split processed data into an eras and non-eras group based on date.
    Abbreviations (E = ERAS and NE = non-ERAS)

    :param raw_pickle: raw data pickle
    :param processed_pickle: cleaned up data pickle
    :param eras_dt: eras implementation datetime

    :returns: eras, non-eras patient lists and eras and non-eras dataframes
    """
    df_raw = pd.read_pickle(raw_pickle) #raw data pickle
    df_processed = pd.read_pickle(processed_pickle) #processed data pickle
    df_raw = df_raw[['patient_id','sx_admission_date_a']] #grabs patient id and sx_admission date (not in processed df)
    df_raw = df_raw[df_raw.sx_admission_date_a.notnull()] #removes null values from raw df

    df_processed = pd.merge(df_processed,df_raw,how='inner',on='patient_id') #inner join to essentially select all patients who have an admission date

    df_E = df_processed[df_processed.sx_admission_date_a>=eras_dt]
    df_NE = df_processed[df_processed.sx_admission_date_a<eras_dt]
    pts_E = df_E.patient_id.tolist()
    pts_NE = df_NE.patient_id.tolist()

    return pts_E, pts_NE, df_E, df_NE

def impute_strategy(df,strategy):
    """
    Imputes valuse based on strategy input. Uses preprocsssing.Imputer() within sklearn

    :param df: dataframe with columns to impute
    :param strategy: i.e. mean and mode

    :returns: imputed dataframe
    """
    fill_NaN = preprocessing.Imputer(missing_values=np.nan, strategy=strategy, axis=0)
    result = pd.DataFrame(fill_NaN.fit_transform(df))
    result.index = df.index
    result.columns = df.columns
    return result

def onehot_data(df):
    """
    This function onehotencodes using preprocessing.OneHotEncoder() within sklearn. Currently there is some hard coding because of differences between groups.

    :param df: dataframe  to be onehot encoded
    :returns: onehotencoded dataframe
    """
    enc = preprocessing.OneHotEncoder()
    for col in df.columns:

        num_of_values = list(df[col].unique())
        num_of_values.sort()
        reshaped_df = df[col].values.reshape(-1,1)
        process = preprocessing.OneHotEncoder()

        #primary_dx 15
        #second_dx 15
        #sex 3
        #These have to hard coded in place because some values do not show up in both sets. Which will allow cross evaluation if needed
        if col=='primary_dx':
            # unique_list = [0.0, 2.0, 3.0, 4.0, 11.0, 10.0, 9.0, 7.0, 16.0, 1.0, 6.0, 5.0]
            process = preprocessing.OneHotEncoder(n_values=18)
            num_of_values = range(18)
        elif col=='second_dx':
            pass
            process = preprocessing.OneHotEncoder(n_values=18)
            num_of_values = range(18)
        elif col=='sex':
            process = preprocessing.OneHotEncoder(n_values=4)
            num_of_values = range(4)
        else:
            pass

        df_onehot = process.fit_transform(reshaped_df)   
        df_onehot = pd.DataFrame(df_onehot.toarray())

        col_list = []

        for i in num_of_values:
            col_list.append('{}_{}'.format(col,i))

        df_onehot.columns = col_list
        df_onehot.index = df.index

        df = pd.concat([df,df_onehot],axis=1)
        df = df.drop(col,1)
      
    return df

def impute(df):
    """
    Imputes missing values. Categories/lists are hardcoded: Missing as value, not missing at random, mean and mode.


    """
    missing_as_value = ['primary_dx','race','second_dx','sex','ethnicity','ho_smoking','sx_diagnosis','sx_facility','surgery_mode'] #set max+1
    not_missing_at_random = ['currenct_medtreatment___14','currenct_medtreatment___15','currenct_medtreatment___16','currenct_medtreatment___17','currenct_medtreatment___18','currenct_medtreatment___19','currenct_medtreatment___20','currenct_medtreatment___21','currenct_medtreatment___22','currenct_medtreatment___23','med_condition___1','med_condition___10','med_condition___11','med_condition___12','med_condition___13','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','cea_value','crp_value','no_ab_sx','no_total_attacks','sx_diversion','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5',] #set to default 0
    impute_mean = ['age','albumin_value','alp_value','bmi','bun_value','creatinine_value','glucose_value','hgb_value','plt_value','prealbumin_value','wbc_value','sx_ebl','sx_length'] #imput mean
    impute_mode = ['asa_class'] #imput mode
    #impute zero: cea_value, crp_value

    output = ['po_sx_readmission','comp_score'] #,'sx_po_stay' removed bc nans and not main question

    #groups Clavien-Dindo groups
    df.comp_score.replace(1,1,inplace=True)    
    df.comp_score.replace(2,2,inplace=True)
    df.comp_score.replace(3,3,inplace=True)        
    df.comp_score.replace(4,3,inplace=True)
    df.comp_score.replace(5,3,inplace=True)

    #unique values of smoking are 14,15,16,17,18,19 #replacing with 0-6 with 6 being nan
    df.ho_smoking.replace(14.,0,inplace=True) #never
    df.ho_smoking.replace(15.,1,inplace=True) #current
    df.ho_smoking.replace(16.,2,inplace=True) #quit <1yr
    df.ho_smoking.replace(17.,3,inplace=True) #quit <5yr
    df.ho_smoking.replace(18.,4,inplace=True) #quit >10yr
    df.ho_smoking.replace(19.,5,inplace=True) #quit

    #redefines asa class values to make sense
    df.asa_class.replace(14.,1,inplace=True)
    df.asa_class.replace(15.,2,inplace=True)
    df.asa_class.replace(16.,3,inplace=True)
    df.asa_class.replace(17.,4,inplace=True)

    #redefenies diversion values to be closer to zero
    df.sx_diversion.replace(17,1,inplace=True) #colostomy
    df.sx_diversion.replace(18,2,inplace=True) #ileostomy

    #Not missing at random default to a value of zero
    for col in not_missing_at_random:
        df[col].fillna(0,inplace=True)

    #Missing at random so set to max+1 value
    for col in missing_as_value:
        df[col].fillna(df[col].max()+1,inplace=True)

    #impute mean and mode for respective groups
    df[impute_mean] = impute_strategy(df[impute_mean],'median')
    df[impute_mode] = impute_strategy(df[impute_mode],'most_frequent')

    df = df.drop(['patient_id','hba1c_value','sx_admission_date_a'],1) #removes extra columns

    df_onehot = onehot_data(df[missing_as_value])
    missing_as_value = df_onehot.columns.tolist()
    df = pd.concat([df,df_onehot],1)
    df_X = df[not_missing_at_random+missing_as_value+impute_mean+impute_mode]
    df_y = df[output]
    # print(df_y.columns[df_y.isnull().any()].tolist())

    X = df_X.as_matrix()
    y_readmit = df_y['po_sx_readmission'].as_matrix()
    y_complication = df_y['comp_score'].as_matrix()
    y = [y_readmit,y_complication]

    return X, y

#f score ranges from 0 to 1
def loocv(X,y,clf):
    """


    """
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    score_list = []
    y_predict = []
    for train_index,test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train,y_train)
        y_predict.append(clf.predict(X_test)[0])
        score_list.append(clf.score(X_test,y_test))
    f1 = f1_score(y,y_predict)
    # print('f1 score: {0:.4f}'.format(f1))
    return f1

def kfoldcv(X,y,clf):
    """

    logic is differnt for kfolds comared to loovc so need to fix this.

    """
    kfold = StratifiedKFold(n_splits=10)
    score_list = []
    y_predict = []
    y_test_list = []
    for train_index,test_index in kfold.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train,y_train)
        y_predict = y_predict + list(clf.predict(X_test))
        y_test_list = y_test_list + list(y_test)
        score_list.append(clf.score(X_test,y_test))
    f1 = f1_score(y_test_list,y_predict)
    # print('f1 score: {0:.4f}'.format(f1))
    return f1

def zerofold(X,y,clf):
    """
    Will do a zero fold cross validation. Used instead of leave one out or kfold cross validation.
    Splits data into 70/30 train/test

    :param X: training inputs
    :param y: training outputs
    :param clf: classifier i.e. decsions tree or svm object

    :returns: 
    """

    X_train, X_test, y_train, y_test = train_test_split(X,y[0],test_size=0.3,random_state=42)
    




def test(x,df_y):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def loop_rec(args):
    """
    Uses the product module within itertools to create a list of tuples that can be passed into another functions parameters. Allos for either lists or dictionaries to be used.

    :param args: can be a value, list, list of lists or a list of dictionaries
    :returns: a list of tuples
    """
    if type(args) == list:
        return list(product(*args))
    elif type(args) == dict:
        return list(product(*args.values())) #need to loop through values



def int_maximize(fxn, args, X, y, clf):
    # result_list = []
    if type(clf) == type(tree.DecisionTreeClassifier()):
        # print('max_depth{}. min_sample{}'.format(args[0],args[1]))
        clf = tree.DecisionTreeClassifier(max_depth=args[0],min_samples_leaf=args[1])
        # result_list.append(fxn(X,y,clf))
        return fxn(X,y,clf)
    elif type(clf) == type(svm.SVC()):
        clf = svm.SVC(C=args[0],kernel='poly',gamma='auto',cache_size=1000,class_weight='balanced')
        return fxn(X,y,clf)        
    else:
        print('Still have not added this classifier')
        return None
    # return max(result_list)


def find_best_parameters(fxn, parameter_dict, X, y, clf):
    parameter_list = loop_rec(parameter_dict)
    f1_score = 0
    f1_params = []
    percentage = 0
    for cnt, parameters in enumerate(parameter_list):
        project_modules.running_fxn(20, percentage, cnt, len(parameter_list))
        new_f1 = int_maximize(fxn, parameters, X, y, clf)
        if new_f1 > f1_score:
            print(new_f1, parameters)
            f1_score = new_f1
            f1_params = parameters


    print('final score: {}\nfinal parameters:{}'.format(f1_score, f1_params))
    return f1_score, f1_params
    
            


# def fxn_to_minimize(x,X,y,clf):
#     # print(x,X,y,clf)
#     if type(clf) == type(tree.DecisionTreeClassifier()):
#         clf = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=int(x[0]))
#         # print(x[0])
#         loocv(X,y,clf)
#     else:
#         pass
    # clf = tree.DecisionTreeClassifier(max_depth=y,min_samples_leaf=x)



def main():
    pts_E, pts_NE, df_E, df_NE = split_eras('S:\ERAS\cr_df.pickle','S:\ERAS\cr_preprocess.pickle',datetime.datetime(2014,7,1,0,0))
    X, y = impute(df_E)
    X_train, X_test, y_train, y_test = train_test_split(X,y[0],test_size=0.3,random_state=42)
    # clf = tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=5)
    # clf = tree.DecisionTreeClassifier()
    clf = svm.SVC()

    # test_list = [[1,2,3],[2,5,2]]
    # test_dict = {'foo':range(1,10),'bar':range(1,10)}
    # print(loop_rec(test_list))

    # print(int_maximize(loocv, None, X_train, y_train,clf))
    tree_parameter_dict = {'max_depth':range(1,10),'min_sample_leaf':range(1,10)}
    svm_parameter_dict = {'c':[0.1]}

    # find_best_parameters(loocv, tree_parameter_dict, X_train, y_train, clf)
    # find_best_parameters(kfoldcv, tree_parameter_dict, X_train, y_train, clf)
    # find_best_parameters(kfoldcv, svm_parameter_dict, X_train, y_train, clf)
    find_best_parameters(loocv,svm_parameter_dict,X_train,y_train,clf)

    # simple_minimize(loocv,{'min_sample_leaf':[1,1000]},X_train,y_train,clf)


    # loocv(X_train,y_train,clf)

    # fxn_to_minimize(1,X,y,clf)
    # result = minimize(fun=fxn_to_minimize,x0=1,args=(X_train,y_train,tree.DecisionTreeClassifier()),bounds=(1,None),method='L-BFGS-B')
    # print(result)
    # x0 = range(0,10)
    # res = minimize(test,x0,method='nelder-mead')
    # print(res)

if __name__ == '__main__':
    main()