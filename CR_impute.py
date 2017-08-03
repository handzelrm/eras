import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, tree, svm
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, average_precision_score
from scipy.optimize import minimize
from itertools import product
import time
import multiprocessing
from scipy import interp
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')

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

    value_dict =  {'race':[0,1,2,3,4,5,6],'primary_dx':[0,1,2,3,4,5,6],'second_dx':[],'sex':[0,1,2],'ethnicity':[0,1,2],'ho_smoking':[0,1,2,3],'surgery_mode':[0,1,2,3,4]}

    # enc = preprocessing.OneHotEncoder()
    for col in df.columns:
        # print(col)
        num_of_values = len(df[col].unique())
        # print(num_of_values)
        # num_of_values = max(df[col].unique())
        num_of_values = len(value_dict[col])
        # num_of_values.sort()
        reshaped_df = df[col].values.reshape(-1,1)
        # print(reshaped_df)
        # process = preprocessing.OneHotEncoder()

        #primary_dx 15
        #second_dx 15
        #sex 3
        #These have to hard coded in place because some values do not show up in both sets. Which will allow cross evaluation if needed
        # if col=='primary_dx':
        #     # unique_list = [0.0, 2.0, 3.0, 4.0, 11.0, 10.0, 9.0, 7.0, 16.0, 1.0, 6.0, 5.0]
        #     process = preprocessing.OneHotEncoder(n_values=18)
        #     num_of_values = range(18)
        # elif col=='second_dx':
        #     pass
        #     process = preprocessing.OneHotEncoder(n_values=18)
        #     num_of_values = range(18)
        # elif col=='sex':
        #     process = preprocessing.OneHotEncoder(n_values=4)
        #     num_of_values = range(4)
        # else:
        #     pass

        # if col=='primary_dx':
        #     # unique_list = [0.0, 2.0, 3.0, 4.0, 11.0, 10.0, 9.0, 7.0, 16.0, 1.0, 6.0, 5.0]
        #     # num_of_values = len(df[col].unique())
        #     # print(num_of_values)
        #     # print(df[col].unique())
        #     process = preprocessing.OneHotEncoder(n_values=num_of_values)
        #     # print(process)
        #     # num_of_values = range(18)
        # elif col=='second_dx':
        #     pass
        #     process = preprocessing.OneHotEncoder(n_values=18)
        #     # num_of_values = range(18)
        # elif col=='sex':
        #     process = preprocessing.OneHotEncoder(n_values=4)
        #     # num_of_values = range(4)
        # else:
        #     pass

        # print(col,num_of_values,df[col].unique())
        # process = preprocessing.OneHotEncoder(n_values=num_of_values)
        process = preprocessing.OneHotEncoder(num_of_values)
        df_onehot = process.fit_transform(reshaped_df)   
        df_onehot = pd.DataFrame(df_onehot.toarray())

        col_list = []

        for i in range(num_of_values):
            col_list.append('{}_{}'.format(col,i))

        # print(col)
        df_onehot.columns = col_list
        df_onehot.index = df.index

        df = pd.concat([df,df_onehot],axis=1)
        df = df.drop(col,1)
      
    return df

def combine_similar_columns(df):
    """
    Will take one hot encoded data and combine all of the columns by taking the max value for each patient.

    :param df: dataframe with all columns that need to consolidated
    :returns: 
    """
    df.sum()

    # return df_out

def impute(df):
    """
    Imputes missing values. Categories/lists are hardcoded: Missing as value, not missing at random, mean and mode.

    :param df: Dataframe to be imputed
    :returns: X, y
    """
    #removed second dx, may need to check if okay
    #removed sx_facility
    #removed sx_diagnosis
    missing_as_value = ['primary_dx','race','sex','ethnicity','ho_smoking','surgery_mode'] #set max+1
    # not_missing_at_random = ['currenct_medtreatment___14','currenct_medtreatment___15','currenct_medtreatment___16','currenct_medtreatment___17','currenct_medtreatment___18','currenct_medtreatment___19','currenct_medtreatment___20','currenct_medtreatment___21','currenct_medtreatment___22','currenct_medtreatment___23','med_condition___1','med_condition___10','med_condition___11','med_condition___12','med_condition___13','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','cea_value','crp_value','no_ab_sx','no_total_attacks','sx_diversion','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5',] #set to default 0
    not_missing_at_random = ['currenct_medtreatment___14','currenct_medtreatment___15','currenct_medtreatment___16','currenct_medtreatment___17','currenct_medtreatment___18','currenct_medtreatment___19','currenct_medtreatment___20','currenct_medtreatment___21','currenct_medtreatment___22','currenct_medtreatment___23','cardiac','renal','copd','diabetes','radiation','hypertension','transplant','cea_value','crp_value','no_ab_sx','no_total_attacks','sx_diversion','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5',] #set to default 0
    
    impute_mean = ['age','albumin_value','alp_value','bmi','bun_value','creatinine_value','glucose_value','hgb_value','plt_value','prealbumin_value','wbc_value','sx_ebl','sx_length'] #imput mean
    impute_mode = ['asa_class'] #imput mode
    #impute zero: cea_value, crp_value

    output = ['po_sx_readmission','comp_score'] #,'sx_po_stay' removed bc nans and not main question

    #Colorectal polpys/cancer
    df.primary_dx.replace(0,0,inplace=True) #rectal cancer
    df.primary_dx.replace(1,0,inplace=True) #rectal polpys
    df.primary_dx.replace(2,0,inplace=True) #colon cancer
    df.primary_dx.replace(3,0,inplace=True) #colon polyps
    df.primary_dx.replace(10,0,inplace=True) #rectal mass
    df.primary_dx.replace(11,0,inplace=True) #colon mass
    df.primary_dx.replace(14,0,inplace=True) #recurrent colon cancer with mets
    df.primary_dx.replace(16,0,inplace=True) #recurrent rectal cancer with mets
    #inflammatrory bowel dz
    df.primary_dx.replace(4,1,inplace=True) #crohns dz
    df.primary_dx.replace(6,1,inplace=True) #ulcerative colitis
    #ischemic colitis
    df.primary_dx.replace(5,2,inplace=True) #ischemic colitis
    #diveticultiis
    df.primary_dx.replace(7,3,inplace=True) #diverticulitis
    #colonic inertia
    df.primary_dx.replace(8,4,inplace=True) #colonic inertia
    df.primary_dx.replace(9,5,inplace=True) #other

    #surgery mode nan will be 4
    df.surgery_mode.replace(1,0,inplace=True) #open
    df.surgery_mode.replace(6,0,inplace=True) #lap converted
    df.surgery_mode.replace(2,1,inplace=True) #hand-assisted
    df.surgery_mode.replace(3,2,inplace=True) #laparoscopic
    df.surgery_mode.replace(4,2,inplace=True) #robotic
    df.surgery_mode.replace(5,2,inplace=True) #laparscopic/robotic
    df.surgery_mode.replace(7,3,inplace=True) #TA TME

    med_cond_dict = {'cardiac':['med_condition___1','med_condition___2','med_condition___3','med_condition___4','med_condition___9','med_condition___11','med_condition___12'],'renal':['med_condition___5'],'copd':['med_condition___6'],'diabetes':['med_condition___7'],'hypertension':['med_condition___8'],'radiation':['med_condition___10'],'transplant':['med_condition___13']}
    
    #loops through dictionary and gets max value for each group (1 if present) and returns array to dictionary which is used to create another dataframe
    for i in med_cond_dict:
        # med_cond_dict[i] = df[med_cond_dict[i]].max(axis=1).values #gets max values in list format
        df[i] = df[med_cond_dict[i]].max(axis=1).values #gets max values in list format
    # med_cond_dict['patient_id'] = list(df.patient_id) #adds patient id for merge
    # temp_df = pd.DataFrame(med_cond_dict) #creates temporary dataframe for merge


    df = df.drop(['med_condition___10', 'med_condition___7', 'med_condition___9', 'med_condition___11', 'med_condition___12', 'med_condition___13', 'med_condition___5', 'med_condition___8', 'med_condition___1', 'med_condition___2', 'med_condition___3', 'med_condition___4', 'med_condition___6'],1) #removes extra columns

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
    df.ho_smoking.replace(17.,2,inplace=True) #quit <5yr
    df.ho_smoking.replace(18.,2,inplace=True) #quit >10yr
    df.ho_smoking.replace(19.,2,inplace=True) #quit

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
        #ethnicity has value of 2 if unknown i.e. nan
        if col == 'ethnicity':
            df[col].fillna(2,inplace=True)
        else:
            df[col].fillna(df[col].max()+1,inplace=True)

    #impute mean and mode for respective groups
    df[impute_mean] = impute_strategy(df[impute_mean],'mean')
    df[impute_mode] = impute_strategy(df[impute_mode],'most_frequent')

    df = df.drop(['patient_id','hba1c_value','sx_admission_date_a'],1) #removes extra columns

    df_onehot = onehot_data(df[missing_as_value])
    missing_as_value = df_onehot.columns.tolist()
    df = pd.concat([df,df_onehot],1)
    df_X = df[not_missing_at_random+missing_as_value+impute_mean+impute_mode]
    df_y = df[output]
    # print(df_y.columns[df_y.isnull().any()].tolist())

    X = df_X.as_matrix()
    # y_readmit = df_y['po_sx_readmission'].as_matrix()
    # y_complication = df_y['comp_score'].as_matrix()
    # y = [y_readmit,y_cocmplication]
    y = df_y.as_matrix()

    return X, y

#f score ranges from 0 to 1
def loocv(X,y,clf):
    """
    Leave one out cross validation. Uses sklearn modules.

    :param X: training input parameters
    :param y: training results
    :param clf: classifer i.e. decision tree object
    :returns: f1 score
    """
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    score_list = []
    y_predict = []
    y_proba = []
    y_proba_test = []
    y_test_list = []
    tpr = 0.0
    fpr = np.linspace(0,1,100)
    # y_proba_1 = []
    # y_proba_2 = []
    for train_index,test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train,y_train)
        y_predict.append(clf.predict(X_test)[0])
        score_list.append(clf.score(X_test,y_test))
        y_proba.append(clf.predict_proba(X_test)[0][0])
        # y_proba = clf.predict_proba(X_test)
        y_test_list.append(y_test)
        # fpr,tpr,_ = roc_curve(y_test,y_proba[:,1])
        # mean_tpr += interp(mean_fpr,fpr,tpr)
        # mean_tpr[0] = 0.0
        # roc_auc = auc(fpr,tpr)
        # y_proba.append(test)
        # print(test)
        # test = clf.predict_proba(X_test)
        # test = np.append(test,clf.predict_proba(X_test))
        # test.append(clf.predict_proba(X_test))
        # print(test)
        # print(clf.predict_proba(X_test)+clf.predict_proba(X_test))
        # return
        # y_proba_1.append(clf.predict_proba(X_test)[0])
        # y_proba_2.append(clf.predict_proba(X_test)[1])


    # print(y_proba_1)
    # mean_tpr /= kfold.get_n_splits(X,y)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr,mean_tpr)
    # print(y_proba)
    # print(len(y_proba))
    # print(len(y_test_list))
    # print(y_test)
    fpr,tpr,_ = roc_curve(y_test_list,y_proba)
    mean_auc = auc(fpr,tpr)
    # print(mean_auc)
    # plot_roc(y_proba_test,y_proba)
    # f1 = f1_score(y,y_predict)
    f1 = f1_score(y,y_predict)
    # print('f1 score: {0:.4f}'.format(f1))
    return f1, mean_auc, fpr, tpr

def kfoldcv(X,y,clf):
    """
    K-fold cross validation. Uses sklearn modules.

    :param X: training input parameters
    :param y: training results
    :param clf: classifer i.e. decision tree object
    :returns: f1 score
    """
    # print(X.shape)
    kfold = StratifiedKFold(n_splits=10)
    score_list = []
    y_predict = []
    y_test_list = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    for train_index,test_index in kfold.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train,y_train)
        y_predict = y_predict + list(clf.predict(X_test))
        y_test_list = y_test_list + list(y_test)
        y_proba = clf.predict_proba(X_test)
        score_list.append(clf.score(X_test,y_test))
        fpr,tpr,_ = roc_curve(y_test,y_proba[:,1])
        mean_tpr += interp(mean_fpr,fpr,tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr,tpr)
        # plt.plot(fpr,tpr)
    # plot_roc(y_test,y_score[:,1])
    mean_tpr /= kfold.get_n_splits(X,y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    # plt.plot(mean_fpr,mean_tpr)
    # plt.show()
    # print(mean_auc)
    f1 = f1_score(y_test_list,y_predict)
    # print('f1 score: {0:.4f}'.format(f1))
    return f1, mean_auc, mean_fpr, mean_tpr

def bag_it():
    bagging = BaggingClassifier()

def zerofold(X,y,clf):
    """
    Will do a zero fold cross validation. Used instead of leave one out or kfold cross validation.
    Splits data into 70/30 train/test

    :param X: training inputs
    :param y: training outputs
    :param clf: classifier i.e. decsions tree or svm object
    :returns: f1 score

    Output the precision recall graph and the ROC graph and send them to me. Feel free to train and test on the full 70% split of your data. Do not worry about cross validation for now. 
    """
    precision = {}
    recall = {}
    average_precision = {}
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    clf = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    # print('score {}'.format(y_score))
    
    for i in [0,1]:
        precision[i], recall[i], _ = precision_recall_curve(y_test,y_score[:,i])
        average_precision[i] = average_precision_score(y_test, y_score[:, i])

    # print('y_test = {}'.format(y_test))
    # print('y_pred = {}'.format(y_predict))
    # plot_recall_precision(recall,precision,average_precision)
    # plot_roc(y_test,y_score[:,1])
    # print(clf.score(X_test,y_test))

    f1 = f1_score(y_test,y_predict)
    
    # plt.clf()
    # plt.plot(precision,recall)
    # plt.show()

    # print(precision,recall,thresholds)
    print(f1)
    return f1


def allfold(X,y,clf):
    """
    Will do a zero fold cross validation. Used instead of leave one out or kfold cross validation.
    Splits data into 70/30 train/test

    :param X: training inputs
    :param y: training outputs
    :param clf: classifier i.e. decsions tree or svm object
    :returns: f1 score

    Output the precision recall graph and the ROC graph and send them to me. Feel free to train and test on the full 70% split of your data. Do not worry about cross validation for now. 
    """
    precision = {}
    recall = {}
    average_precision = {}
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    clf = clf.fit(X,y)
    y_predict = clf.predict(X)
    y_score = clf.predict_proba(X)
    # print('score {}'.format(y_score))
    
    for i in [0,1]:
        precision[i], recall[i], _ = precision_recall_curve(y,y_score[:,i])
        average_precision[i] = average_precision_score(y, y_score[:, i])

    print('y = {}'.format(y))
    print('y_pred = {}'.format(y_predict))
    # plot_recall_precision(recall,precision,average_precision)
    plot_roc(y,y_score[:,1])
    # print(clf.score(X_test,y_test))

    f1 = f1_score(y,y_predict)
    
    # plt.clf()
    # plt.plot(precision,recall)
    # plt.show()

    # print(precision,recall,thresholds)
    print(f1)
    return f1


def plot_recall_precision(recall,precision,average_precision):
    """
    Will plot the precision/recall graph and report the AUC. Loops through each of the dictionary items.

    :param recall: can be a dictionary
    :param precisison: can be a dictionary
    :param precision: can be a dictionary
    :returns: precision/recall plot
    """
    
    for i in recall:

        plt.clf() #clears figure
        plt.plot(recall[i], precision[i], color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[i]))
        plt.legend(loc="lower left")
        plt.show()

def plot_roc(y,y_scores):
    """
    Will plot the ROC curve. Loops through each of the dictionary items.

    :param y: can be a dictionary
    :param y_score: can be a dictionary
    :returns: ROC curve
    """
    fpr, tpr, _ = roc_curve(y, y_scores, pos_label=1)
    plt.clf()
    plt.plot(fpr,tpr, color='navy',
             label='ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[i]))
    plt.legend(loc="lower left")
    plt.show()

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
    """


    :returns: result of the function that was inputed
    """
    # result_list = []
    if type(clf) == type(tree.DecisionTreeClassifier()):
        # print('max_depth{}. min_sample{}'.format(args[0],args[1]))
        clf = tree.DecisionTreeClassifier(max_depth=args[0],min_samples_leaf=args[1])
        # result_list.append(fxn(X,y,clf))
        return fxn(X,y,clf)
    elif type(clf) == type(svm.SVC()):
        clf = svm.SVC(C=args[0],kernel='poly',gamma='auto',cache_size=1000,class_weight='balanced')
        return fxn(X,y,clf)
    elif type(clf) == type(RandomForestClassifier()):
        clf = RandomForestClassifier(n_estimators=args[0],max_features=args[1],max_depth=None,min_samples_split=2,random_state=0,n_jobs=-1)
        return fxn(X,y,clf)
    else:
        print('Still have not added this classifier')
        return None
    # return max(result_list)


def find_best_parameters(fxn, parameter_dict, X, y, clf):
    """


    """
    parameter_list = loop_rec(parameter_dict)
    f1_score = 0
    best_auc = 0.5
    f1_params = []
    auc_params = []
    percentage = 0
    for cnt, parameters in enumerate(parameter_list):
        project_modules.running_fxn(20, percentage, cnt, len(parameter_list))
        new_f1,new_auc,fpr,tpr = int_maximize(fxn, parameters, X, y, clf)
        # print(new_f1)
        if new_f1 > f1_score:
            # print(new_f1, parameters)
            f1_score = new_f1
            f1_params = parameters
        # print(abs(new_auc-0.5),abs(best_auc-0.5))
        if abs(new_auc-0.5) > abs(best_auc-0.5):
            print(new_auc, parameters)
            best_auc = new_auc
            auc_params = parameters
            plt.plot(fpr,tpr)
            plt.title('AUC:{}, Parameters{}'.format(new_auc,parameters))
            plt.show()


    if best_auc < 0.5:
        best_auc += 0.5
    print('max f1 score: {}\nparameters:{}'.format(f1_score, f1_params))
    print('max AUC: {}\nparameters:{}'.format(best_auc, auc_params))
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
    """


    """
    pts_E, pts_NE, df_E, df_NE = split_eras('S:\ERAS\cr_df.pickle','S:\ERAS\cr_preprocess.pickle',datetime.datetime(2014,7,1,0,0))
    X, y = impute(df_E)
    # print('X{}, y{}'.format(X.shape,y.shape))
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y[:,0],test_size=0.3,random_state=42) #y[0] = readmits
    # print('Train{}, Test{}'.format(X_train.shape,X_test.shape))
    # print(109/(109+256))
    # clf = tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=5)
    # clf = svm.SVC()

    # test_list = [[1,2,3],[2,5,2]]
    # test_dict = {'foo':range(1,10),'bar':range(1,10)}
    # print(loop_rec(test_list))

    # print(int_maximize(loocv, None, X_train, y_train,clf))
    tree_parameter_dict = {'max_depth':range(2,15),'min_sample_leaf':range(2,15)}
    forest_parameter_dict = {'n_estimators':[1000],'max_features':[18]}
    svm_parameter_dict = {'c':[0.1]}
    # print(y_train)
    # find_best_parameters(kfoldcv, tree_parameter_dict, X_train, y_train, tree.DecisionTreeClassifier())
    # find_best_parameters(loocv, tree_parameter_dict, X_train, y_train, tree.DecisionTreeClassifier())
    # find_best_parameters(loocv, forest_parameter_dict, X_train, y_train, RandomForestClassifier())
    find_best_parameters(kfoldcv, forest_parameter_dict, X_train, y_train, RandomForestClassifier())

    # find_best_parameters(kfoldcv, svm_parameter_dict, X_train, y_train, clf)
    # find_best_parameters(loocv,svm_parameter_dict,X_train,y_train,clf)
    # find_best_parameters(zerofold,tree_parameter_dict,X_train,y_train,clf)
    # find_best_parameters(allfold,tree_parameter_dict,X_train,y_train,clf)
    # find_best_parameters(kfoldcv,forest_parameter_dict, X_train, y_train,RandomForestClassifier())

    # simple_minimize(loocv,{'min_sample_leaf':[1,1000]},X_train,y_train,clf)


    # loocv(X_train,y_train,clf)

    # fxn_to_minimize(1,X,y,clf)
    # result = minimize(fun=fxn_to_minimize,x0=1,args=(X_train,y_train,tree.DecisionTreeClassifier()),bounds=(1,None),method='L-BFGS-B')
    # print(result)
    # x0 = range(0,10)
    # res = minimize(test,x0,method='nelder-mead')
    # print(res)


    """
    1-15 for both and 10 fold
    max f1 score of 0.2 with parameters of 4,6
    max auc of 0.7238 with paremeters of 5,7



    """

if __name__ == '__main__':
    main()