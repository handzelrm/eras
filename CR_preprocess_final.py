import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing, tree, datasets, metrics, svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import os
import pydotplus
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')
from scipy import interp
from itertools import cycle

def split_eras():
    df = pd.read_pickle('S:\ERAS\cr_df.pickle')
    df_all = pd.read_pickle('S:\ERAS\cr_preprocess.pickle')
    # print('df_all shape: {}'.format(df_all.shape))
    df = df[['patient_id','sx_admission_date_a']]
    df = df[df.sx_admission_date_a.notnull()]
    # print(df[pd.isnull(df.sx_admission_date_a)].shape)
    # print(df.shape)

    df_all = pd.merge(df_all,df,how='inner',on='patient_id')
    df_all = df_all[pd.notnull(df_all.sx_admission_date_a)]
    eras_dt = datetime.datetime(2014,7,1,0,0) #date at which ERAS began


    #plotting out data distributions
    #reruns is error and stores columns with error in pickle file
    # ignore = []
    # error = False
    # for i in df_all.columns:        
    #     try:
    #         ignore = list(pd.read_pickle('S:\ERAS\ignore_plt.pickle'))
    #     except:
    #         error = True
    #     if i not in ignore:
    #         try:
    #             n, bins, patches = plt.hist(df_all[i].dropna())
    #             plt.axvline(df_all[i].dropna().mean(),color='b',linestyle='dashed',linewidth=2)
    #             plt.axvline(df_all[i].dropna().median(),color='g',linestyle='dashed',linewidth=2)
    #             plt.title(i)
    #             plt.show()
    #         except:
    #             error = True
    #             ignore.append(i)
    # if error == True:
    #     pd.to_pickle(ignore,'S:\ERAS\ignore_plt.pickle')
    #     split_eras()
    

    eras = df_all[df_all.sx_admission_date_a>=eras_dt]
    non_eras = df_all[df_all.sx_admission_date_a<eras_dt]

    # print(non_eras.head())
    dups = [70,647,700,1473]

    # print(eras[eras.patient_id.isin(dups)].shape)
    # print(non_eras[non_eras.patient_id.isin(dups)].shape)

    # print('all:{}, eras:{}, non_eras:{}'.format(df_all.shape,eras.shape,non_eras.shape))

    # print(eras.head())

    pd.to_pickle(eras,'S:\ERAS\cr_eras.pickle')
    pd.to_pickle(non_eras,'S:\ERAS\cr_non_eras.pickle')

    eras_pts = list(eras.patient_id)
    non_eras_pts = list(non_eras.patient_id)

    return eras_pts, non_eras_pts

eras_pts, non_eras_pts = split_eras()


def impute_df(strat,df):
    fill_NaN = preprocessing.Imputer(missing_values=np.nan, strategy=strat, axis=0)
    result = pd.DataFrame(fill_NaN.fit_transform(df))
    result.index = df.index
    result.columns = df.columns
    return result

#primary_dx 15
#second_dx 15
#sex 3
def reg_data(process,df):
    for col in df.columns:

        num_of_values = list(df[col].unique())
        num_of_values.sort()
        reshaped_df = df[col].values.reshape(-1,1)
        process = preprocessing.OneHotEncoder()
        
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

            # print(df_reshaped_df.shape)
            # print(df_reshaped_df)
        df_onehot = process.fit_transform(reshaped_df)

            # print(df[df.isnull()])
       
        df_onehot = pd.DataFrame(df_onehot.toarray())

        col_list = []

        for i in num_of_values:
            col_list.append('{}_{}'.format(col,i))

        df_onehot.columns = col_list
        df_onehot.index = df.index

        df = pd.concat([df,df_onehot],axis=1)
        df = df.drop(col,1)
      
    return df

def impute_data(df):
    missing_as_value = ['primary_dx','race','second_dx','sex','ethnicity','ho_smoking','sx_diagnosis','sx_facility','surgery_mode'] #set max+1
    not_missing_at_random = ['currenct_medtreatment___14','currenct_medtreatment___15','currenct_medtreatment___16','currenct_medtreatment___17','currenct_medtreatment___18','currenct_medtreatment___19','currenct_medtreatment___20','currenct_medtreatment___21','currenct_medtreatment___22','currenct_medtreatment___23','med_condition___1','med_condition___10','med_condition___11','med_condition___12','med_condition___13','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','cea_value','crp_value','no_ab_sx','no_total_attacks','sx_diversion','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5',] #set to default 0
    impute_mean = ['age','albumin_value','alp_value','bmi','bun_value','creatinine_value','glucose_value','hgb_value','plt_value','prealbumin_value','wbc_value','sx_ebl','sx_length'] #imput mean
    impute_mode = ['sx_score','asa_class'] #imput mode
    #impute zero: cea_value, crp_value

    output = ['po_sx_readmission','sx_po_stay','comp_score']    

    # print(df[missing_as_value].head())

    df.comp_score.replace(1,1,inplace=True)    
    df.comp_score.replace(2,1,inplace=True)
    df.comp_score.replace(3,2,inplace=True)        
    df.comp_score.replace(4,3,inplace=True)
    df.comp_score.replace(5,3,inplace=True)



    #unique values of smoking are 14,15,16,17,18,19 #replacing with 0-6 with 6 being nan
    df.ho_smoking.replace(14.,0,inplace=True) #never
    df.ho_smoking.replace(15.,1,inplace=True) #current
    df.ho_smoking.replace(16.,2,inplace=True) #quit <1yr
    df.ho_smoking.replace(17.,3,inplace=True) #quit <5yr
    df.ho_smoking.replace(18.,4,inplace=True) #quit >10yr
    df.ho_smoking.replace(19.,5,inplace=True) #quit
    df.ho_smoking.fillna(6,inplace=True)

    df.asa_class.replace(14.,1,inplace=True)
    df.asa_class.replace(15.,2,inplace=True)
    df.asa_class.replace(16.,3,inplace=True)
    df.asa_class.replace(17.,4,inplace=True)

    df.sx_diversion.replace(17,1,inplace=True) #colostomy
    df.sx_diversion.replace(18,2,inplace=True) #ileostomy

    # print(not_missing_at_random)
    for col in not_missing_at_random:
        df[col].fillna(0,inplace=True)
    # df.cea_value.fillna(0,inplace=True)
    # print(df[not_missing_at_random])
    # df[not_missing_at_random].fillna(0,inplace=True)


    df.primary_dx.fillna(df.primary_dx.max()+1,inplace=True)
    df.second_dx.fillna(df.second_dx.max()+1,inplace=True)
    df.race.fillna(df.race.max()+1,inplace=True)
    df.ethnicity.fillna(df.ethnicity.max()+1,inplace=True)
    df.sex.fillna(df.sex.max()+1,inplace=True)

    df[impute_mean] = impute_df('median',df[impute_mean])
    df[impute_mode] = impute_df('most_frequent',df[impute_mode])

    df = df.drop(['patient_id','hba1c_value','sx_admission_date_a'],1) #removes extra rows

    # test = df[missing_as_value]
    # for col in test:
    #     nana = test[col][test[col].isnull()]
    #     print(col,nana.shape)

    # print(test.sx_facility.unique())
    # print(test[test.surgery_mode.isnull()].shape)
    # print(test)
    # print(test.shape)
    # print(test.surgery_mode.unique())
    df.surgery_mode.fillna(8,inplace=True)
    df.sx_facility.fillna(3,inplace=True)


    # print(df.cea_value)

    enc = preprocessing.OneHotEncoder()
    # print(df[missing_as_value].head())
    df_reg = reg_data(enc,df[missing_as_value])

    missing_as_value = list(df_reg.columns)
    # print(missing_as_value)
    return
    df = pd.concat([df,df_reg],1)

    df_input = df[not_missing_at_random+missing_as_value+impute_mean+impute_mode]
    df_output = df[output]

    df_comp = df[df.comp_score>0]
    df_readmit = df[df.po_sx_readmission>0]

    X_comp = df_comp[not_missing_at_random+missing_as_value+impute_mean+impute_mode].as_matrix()
    y_only_comp = df_comp.comp_score.as_matrix()

    X_readmit = df_readmit[not_missing_at_random+missing_as_value+impute_mean+impute_mode].as_matrix()
    y_only_readmit = df_readmit.po_sx_readmission.as_matrix()

    X = df_input.as_matrix()
    y_readmit = df_output.po_sx_readmission.as_matrix()
    y_los = df_output.sx_po_stay.as_matrix()
    y_comp = df_output.comp_score.as_matrix()
    y_cols = list(df_input.columns)

    return X, y_readmit, y_los, y_comp, y_cols, X_comp, y_only_comp, X_readmit, y_only_readmit

def cross_validate(max_depth,min_samples_leaf,X,y,multiclass,group_name):
    splits = 10
    cv = StratifiedKFold(n_splits=splits)
    # classifier = svm.SVC(probability=True)
    # classifier = RandomForestClassifier()
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','black','red','magenta','green'])
    lw = 2
    i = 0
    accuracy = []
    predict_list = []
    y_list = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    n_classes = np.unique(y).shape[0]
 
    cd_names = ['No Complication', 'Clavien-Dindo I', 'Clavien-Dindo II', 'Clavien-Dindo III', 'Clavien-Dindo IV', 'Clavien-Dindo V'] #hardcode for complication score names for plot titles
    fpr_class = [[],[],[],[],[],[]]
    tpr_class = [[],[],[],[],[],[]]
    auc_class = [[],[],[],[],[],[]]

    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        classPredicted = classifier.fit(X[train], y[train]).predict(X[test])
        predict_list.extend(classPredicted)

        if not multiclass:
            fpr, tpr, thresholds = roc_curve(y[test],probas_[:,1])
            mean_tpr +=interp(mean_fpr,fpr,tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr,tpr)
            plt.plot(fpr,tpr,lw=lw, color=color, label = 'ROC fold {} (area={})'.format(i,round(roc_auc,2)))
        
        elif multiclass:
            class_roc_auc = []
            class_fpr = []
            class_tpr = []
            for j in range(n_classes):
                fpr, tpr, thresholds = roc_curve(y[test],probas_[:,1],pos_label=j)
                roc_auc = auc(fpr,tpr) #calculates auc

                fpr_class[j].append(fpr)
                tpr_class[j].append(tpr)
                auc_class[j].append(roc_auc)

        i += 1
        y_list.extend(y[test])
        accuracy.append(metrics.accuracy_score(y[test], classPredicted))   

    if multiclass:
        #loops through each of the classes
        for j in range(n_classes):
            mean_tpr = 0.0
            mean_fpr = np.linspace(0,1,100)
            #loops through each of the cv which is nested within each class
            for i,color in zip(range(splits),colors):
                plt.plot(fpr_class[j][i],tpr_class[j][i],lw=lw, color=color, label = 'ROC fold {} (area={})'.format(i,round(auc_class[j][i],2))) #plots a single cv roc
                mean_tpr += interp(mean_fpr,fpr_class[j][i],tpr_class[j][i])
            mean_tpr[0] = 0.0
            mean_tpr /= 10
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = {})'.format(round(mean_auc,2)), lw=lw)
            plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Luck')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate', fontsize=28)
            plt.ylabel('True Positive Rate', fontsize=28)
            plt.title('{} Complications ({}) ROC Curve'.format(group_name,cd_names[j]), fontsize=28)
            plt.legend(loc="lower right", fontsize=24)
            plt.show()
            print('\t{} Mean ROC (area = {})'.format(cd_names[j],round(mean_auc,2)))
    

    if not multiclass:
        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = {})'.format(round(mean_auc,2)), lw=lw)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=28)
        plt.ylabel('True Positive Rate', fontsize=28)
        plt.title('{} Readmission ROC Curve'.format(group_name), fontsize=28)
        plt.legend(loc="lower right", fontsize=24)
        plt.show()
        print('\tMean ROC (area = {})'.format(round(mean_auc,2)))
    
    # print('\tAccuracy: {}%'.format(round(100*np.mean(accuracy),2)))

    return accuracy

def build_a_tree(max_depth,min_samples_leaf,X,y,features,file_out):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    clf = clf.fit(X,y)
    dot_data = tree.export_graphviz(clf,out_file=None,feature_names=features,filled=False,rounded=True,proportion=True, rotate=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(file_out)

def compare_groups(max_depth, min_sample_leaf, X_eras, y_eras, X_non, y_non, eras_X_data, eras_y_data, non_X_data, non_y_data):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_sample_leaf)
    eras_clf = clf.fit(X_eras,y_eras)
    non_clf = clf.fit(X_non,y_non)

    eras_clf_predicted = eras_clf.predict(non_X_data)
    df_non_eras_y = np.column_stack([non_y_data,eras_clf_predicted])
    df_non_eras_y = pd.DataFrame(df_non_eras_y)
    df_non_eras_y.columns = ['non_eras_y','eras_predicted_y']
    num_of_eras_predicted_zeros = df_non_eras_y.eras_predicted_y[df_non_eras_y.eras_predicted_y==0].count()
    print('\teras prediction: \t\t{}/{}\t\t{}%'.format(num_of_eras_predicted_zeros,non_y_data.shape[0],round(100*num_of_eras_predicted_zeros/non_y_data.shape[0],2)))

    non_clf_predicted = non_clf.predict(eras_X_data)
    df_eras_y = np.column_stack([eras_y_data,non_clf_predicted])
    df_eras_y = pd.DataFrame(df_eras_y)
    df_eras_y.columns = ['eras_y','non_eras_predicted_y']
    num_of_non_eras_predicted_zeros = df_eras_y.non_eras_predicted_y[df_eras_y.non_eras_predicted_y==0].count()
    print('\tnon eras prediction: \t{}/{}\t\t{}%'.format(num_of_non_eras_predicted_zeros,eras_y_data.shape[0],round(100*num_of_non_eras_predicted_zeros/eras_y_data.shape[0],2)))

df_eras = pd.read_pickle('S:\ERAS\cr_eras.pickle')
df_non_eras = pd.read_pickle('S:\ERAS\cr_non_eras.pickle')


#runs imputation for eras and non-eras

eras_X, eras_y_readmit, eras_y_los, eras_y_comp, eras_cols, eras_X_comp, eras_y_only_comp, eras_X_readmit, eras_y_only_readmit = impute_data(df_eras)
non_eras_X, non_eras_y_readmit, non_eras_y_los, non_eras_y_comp, non_eras_cols, non_eras_X_comp, non_eras_y_only_comp, non_eras_X_readmit, non_eras_y_only_readmit = impute_data(df_non_eras)

#sets the max_depth and min sample leaves for all functions
max_depth = 5

min_samples_leaf = 20

#runs cross model comparison and prints out results
print('complications')
# compare_groups(max_depth, min_samples_leaf, eras_X, eras_y_comp, non_eras_X, non_eras_y_comp, eras_X_comp, eras_y_only_comp, non_eras_X_comp, non_eras_y_only_comp)
print('readmissions')
# compare_groups(max_depth, min_samples_leaf, eras_X, eras_y_readmit, non_eras_X, non_eras_y_readmit, eras_X_readmit, eras_y_only_readmit, non_eras_X_readmit, non_eras_y_only_readmit)

#runs cross validaotin function and creates ROC curves with AUC
print('ERAS Readmissions')
cv_eras_readmit = cross_validate(max_depth,min_samples_leaf,eras_X,eras_y_readmit,multiclass=False,group_name='ERAS')
print('ERAS Complications')
cv_eras_comp = cross_validate(max_depth,min_samples_leaf,eras_X,eras_y_comp,multiclass=True,group_name='ERAS')
print('Non-ERAS Readmissions')
cv_non_eras_readmit = cross_validate(max_depth,min_samples_leaf,non_eras_X,non_eras_y_readmit,multiclass=False,group_name='Non-ERAS')
print('Non-ERAS Complications')
cv_non_eras_comp = cross_validate(max_depth,min_samples_leaf,non_eras_X,non_eras_y_comp,multiclass=True,group_name='Non-ERAS')

#runs build a tree function, which uses the decision tree classifer to build trees for each of the models
build_a_tree(max_depth,min_samples_leaf,eras_X,eras_y_readmit,eras_cols,'DT_ERAS_readmit.pdf')
build_a_tree(max_depth,min_samples_leaf,eras_X,eras_y_comp,eras_cols,'DT_ERAS_comp.pdf')
build_a_tree(max_depth,min_samples_leaf,non_eras_X,non_eras_y_readmit,non_eras_cols,'DT_non_eras_readmit.pdf')
build_a_tree(max_depth,min_samples_leaf,non_eras_X,non_eras_y_comp,non_eras_cols,'DT_non_eras_comp.pdf')