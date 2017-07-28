import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use('ggplot')
import re

#used to add parent directory and reload module
import os
parent_dir = os.path.join(os.getcwd(),os.pardir)
import sys
sys.path.append(parent_dir)
import project_modules
import importlib
importlib.reload(project_modules)

def cr_columns(raw_data):
    """
    Takes in the all data and filters based on columns needed

    :param raw_data: raw pandas dataframe
    :returns: returns filtered dataframe
    """
    df=raw_data[['patient_id','redcap_event_name','age','sex','race','ethnicity','hospital','bmi','primary_dx','other_dx_','other_dx_','second_dx','other_second_dx','pt_hx_statusdiv___1','pt_hx_statusdiv___2','pt_hx_statusdiv___3','no_compl_attacks','no_divattacks_hospital','no_total_attacks','no_ab_sx','prior_ab_sx___0','prior_ab_sx___1','prior_ab_sx___2','prior_ab_sx___3','prior_ab_sx___4','prior_ab_sx___5','prior_ab_sx___6','prior_ab_sx___7','prior_ab_sx___8','prior_ab_sx___9','prior_ab_sx___10','prior_ab_sx___11','prior_ab_sx___12','prior_ab_sx___13','prior_ab_sx___14','prior_ab_sx___15','prior_ab_sx___16','prior_ab_sx___17','prior_ab_sx___18','prior_ab_sx___19','prior_ab_sx_other','med_condition___1','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','med_condition___10','med_condition___11','med_condition___12','med_condition___13','current_medtreatment___14','current_medtreatment___15','current_medtreatment___16','current_medtreatment___17','current_medtreatment___18','current_medtreatment___19','current_medtreatment___20','current_medtreatment___21','current_medtreatment___22','current_medtreatment___23','asa_class','ho_smoking','eval_dx_a','cea_value','wbc_value','hgb_value','plt_value','bun_value','creatinine_value','albumin_value','alp_value','glucose_value','hba1c_value','prealbumin_value','crp_value',
    'sx_diagnosis_a','sx_admission_date_a','sx_date_a','sx_discharge_date_a','sx_po_stay_a','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5','sx_facility_a','sx_urgency_a','surgery_mode_a','prim_sx_rectalca_a___7','prim_sx_rectalca_a___8','prim_sx_rectalca_a___9','prim_sx_rectalca_a___10','prim_sx_rectalca_a___25','prim_sx_rectalca_a___11','prim_sx_rectalca_a___31','prim_sx_rectalca_a___30','prim_sx_rectalca_a___13','prim_sx_rectalca_a___14','prim_sx_rectalca_a___15','prim_sx_rectalca_a___27','prim_sx_rectalca_a___24','prim_sx_rectalca_a___16','prim_sx_rectalca_a___17','prim_sx_rectalca_a___18','prim_sx_rectalca_a___19','prim_sx_rectalca_a___28','prim_sx_rectalca_a___29','prim_sx_rectalca_a___20','prim_sx_rectalca_a___21','prim_sx_rectalca_a___22','prim_sx_rectalca_a___23','prim_sx_rectalpolyp_a___7','prim_sx_rectalpolyp_a___8','prim_sx_rectalpolyp_a___9','prim_sx_rectalpolyp_a___10','prim_sx_rectalpolyp_a___25','prim_sx_rectalpolyp_a___11','prim_sx_rectalpolyp_a___12','prim_sx_rectalpolyp_a___30','prim_sx_rectalpolyp_a___29','prim_sx_rectalpolyp_a___13','prim_sx_rectalpolyp_a___26','prim_sx_rectalpolyp_a___14','prim_sx_rectalpolyp_a___15','prim_sx_rectalpolyp_a___16','prim_sx_rectalpolyp_a___24','prim_sx_rectalpolyp_a___27','prim_sx_rectalpolyp_a___17','prim_sx_rectalpolyp_a___18','prim_sx_rectalpolyp_a___19','prim_sx_rectalpolyp_a___28','prim_sx_rectalpolyp_a___20','prim_sx_rectalpolyp_a___21','prim_sx_rectalpolyp_a___22','prim_sx_rectalpolyp_a___23','prim_sx_other_rectalca_a','prim_sx_other_rectlpolyp_a','prim_sx_colonca_a___7','prim_sx_colonca_a___8','prim_sx_colonca_a___9','prim_sx_colonca_a___10','prim_sx_colonca_a___11','prim_sx_colonca_a___12','prim_sx_colonca_a___32','prim_sx_colonca_a___13','prim_sx_colonca_a___14','prim_sx_colonca_a___15','prim_sx_colonca_a___16','prim_sx_colonca_a___35','prim_sx_colonca_a___36','prim_sx_colonca_a___34','prim_sx_colonca_a___29','prim_sx_colonca_a___28','prim_sx_colonca_a___17','prim_sx_colonca_a___18','prim_sx_colonca_a___19','prim_sx_colonca_a___27','prim_sx_colonca_a___20','prim_sx_colonca_a___30','prim_sx_colonca_a___21','prim_sx_colonca_a___22','prim_sx_colonca_a___31','prim_sx_colonca_a___23','prim_sx_colonca_a___24','prim_sx_colonca_a___25','prim_sx_colonca_a___26','prim_sx_colonpolyp_a___7','prim_sx_colonpolyp_a___8','prim_sx_colonpolyp_a___9','prim_sx_colonpolyp_a___10','prim_sx_colonpolyp_a___11','prim_sx_colonpolyp_a___12','prim_sx_colonpolyp_a___32','prim_sx_colonpolyp_a___13','prim_sx_colonpolyp_a___14','prim_sx_colonpolyp_a___15','prim_sx_colonpolyp_a___16','prim_sx_colonpolyp_a___33','prim_sx_colonpolyp_a___34','prim_sx_colonpolyp_a___35','prim_sx_colonpolyp_a___29','prim_sx_colonpolyp_a___28','prim_sx_colonpolyp_a___17','prim_sx_colonpolyp_a___18','prim_sx_colonpolyp_a___19','prim_sx_colonpolyp_a___20','prim_sx_colonpolyp_a___30','prim_sx_colonpolyp_a___21','prim_sx_colonpolyp_a___27','prim_sx_colonpolyp_a___22','prim_sx_colonpolyp_a___31','prim_sx_colonpolyp_a___23','prim_sx_colonpolyp_a___24','prim_sx_colonpolyp_a___25','prim_sx_colonpolyp_a___26','prim_sx_other_colonca_a','prim_sx_other_colonpolyp_a','prim_sx_bencolon_a___1','prim_sx_bencolon_a___2','prim_sx_bencolon_a___3','prim_sx_bencolon_a___4','prim_sx_bencolon_a___5','prim_sx_bencolon_a___6','prim_sx_bencolon_a___7','prim_sx_bencolon_a___8','prim_sx_bencolon_a___9','prim_sx_bencolon_a___10','prim_sx_bencolon_a___11','prim_sx_bencolon_a___12','prim_sx_bencolon_a___13','prim_sx_bencolon_a___14','prim_sx_bencolon_a___15','prim_sx_bencolon_a___16','prim_sx_bencolon_a___25','prim_sx_bencolon_a___17','prim_sx_bencolon_a___18','prim_sx_bencolon_a___24','prim_sx_bencolon_a___19','prim_sx_bencolon_a___26','prim_sx_bencolon_a___27','prim_sx_bencolon_a___20','prim_sx_bencolon_a___21','prim_sx_bencolon_a___22','prim_sx_bencolon_a___23','prim_sx_other_bencolon_a','sx_rectopexy_a','prim_sx_uc_a___1','prim_sx_uc_a___2','prim_sx_uc_a___31','prim_sx_uc_a___3','prim_sx_uc_a___4','prim_sx_uc_a___5','prim_sx_uc_a___24','prim_sx_uc_a___25','prim_sx_uc_a___6','prim_sx_uc_a___7','prim_sx_uc_a___8','prim_sx_uc_a___29','prim_sx_uc_a___9','prim_sx_uc_a___10','prim_sx_uc_a___11','prim_sx_uc_a___22','prim_sx_uc_a___23','prim_sx_uc_a___12','prim_sx_uc_a___13','prim_sx_uc_a___14','prim_sx_uc_a___15','prim_sx_uc_a___26','prim_sx_uc_a___16','prim_sx_uc_a___21','prim_sx_uc_a___27','prim_sx_uc_a___28','prim_sx_uc_a___17','prim_sx_uc_a___18','prim_sx_uc_a___19','prim_sx_uc_a___20','prim_sx_ic_a___1','prim_sx_ic_a___2','prim_sx_ic_a___30','prim_sx_ic_a___31','prim_sx_ic_a___3','prim_sx_ic_a___4','prim_sx_ic_a___24','prim_sx_ic_a___25','prim_sx_ic_a___5','prim_sx_ic_a___6','prim_sx_ic_a___7','prim_sx_ic_a___8','prim_sx_ic_a___29','prim_sx_ic_a___9','prim_sx_ic_a___10','prim_sx_ic_a___11','prim_sx_ic_a___22','prim_sx_ic_a___23','prim_sx_ic_a___12','prim_sx_ic_a___13','prim_sx_ic_a___14','prim_sx_ic_a___15','prim_sx_ic_a___26','prim_sx_ic_a___16','prim_sx_ic_a___21','prim_sx_ic_a___27','prim_sx_ic_a___28','prim_sx_ic_a___17','prim_sx_ic_a___18','prim_sx_ic_a___19','prim_sx_ic_a___20','prim_sx_cd_a___1','prim_sx_cd_a___2','prim_sx_cd_a___30','prim_sx_cd_a___31','prim_sx_cd_a___3','prim_sx_cd_a___4','prim_sx_cd_a___24','prim_sx_cd_a___25','prim_sx_cd_a___5','prim_sx_cd_a___6','prim_sx_cd_a___7','prim_sx_cd_a___8','prim_sx_cd_a___29','prim_sx_cd_a___9','prim_sx_cd_a___10','prim_sx_cd_a___11','prim_sx_cd_a___22','prim_sx_cd_a___23','prim_sx_cd_a___26','prim_sx_cd_a___12','prim_sx_cd_a___13','prim_sx_cd_a___14','prim_sx_cd_a___15','prim_sx_cd_a___16','prim_sx_cd_a___21','prim_sx_cd_a___27','prim_sx_cd_a___28','prim_sx_cd_a___17','prim_sx_cd_a___18','prim_sx_cd_a___19','prim_sx_cd_a___20','prim_sx_other_uc_a','prim_sx_other_ic_a','prim_sx_other_cd_a','sx_multivisc_rxn_a','sx_anastomosis_a','sx_anastamosis_ibd_a','sx_temp_diversion_a','secondary_sx_a___17','secondary_sx_a___18','secondary_sx_a___19','secondary_sx_a___20','secondary_sx_a___21','secondary_sx_a___22','secondary_sx_a___23','secondary_sx_a___24','secondary_sx_a___25','secondary_sx_a___26','secondary_sx_a___27','secondary_sx_a___28','secondary_sx_a___29','secondary_sx_a___30','other_secondary_sx_a','sx_comb_service_a___16','sx_comb_service_a___17','sx_comb_service_a___18','sx_comb_service_a___19','sx_comb_service_a___20','sx_comb_service_a___21','sx_ebl_a','sx_length_a',
    'post_op_compl_a_dx','po_complication_a___1','po_complication_a___2','po_complication_a___3','po_complication_a___16','po_complication_a___4','po_complication_a___5','po_complication_a___6','po_complication_a___7','po_complication_a___8','po_complication_a___9','po_complication_a___17','po_complication_a___15','po_complication_a___13','po_complication_a___10','po_complication_a___14','po_complication_a___11','po_complication_a___12','po_complication_sx_a','po_leak_repair_a','po_sx_bleeding_repair','po_sx_bowel_obstrctn_a','po_sx_dpwoundinfection_a','po_sx_entericfistula_a','po_sx_dehiscence_a','po_sx_hemorrhage_a','po_sx_hernia_a','po_sx_ischemia_a','po_sx_intraab_infection_a','po_sx_intraab_bleed_a','po_sx_readmission_a','po_sx_superficialwound_a','po_sx_urinary_dysfnctn_a','comments_po_compl_a','po_med_complication_a___1','po_med_complication_a___2','po_med_complication_a___3','po_med_complication_a___4','po_med_complication_a___6','po_med_complication_a___5','po_compl_baselinecr_a','po_compl_elevatedcr_a','po_compl_arf_a','po_medcompl_afib_a','po_other_medcompl_a','po_compl_death_a','po_compl_dod_a','po_compl_cod_a','post_op_complications_a_complete',
    ]]
    return df

def pickle_surgeries():
    """
    Takes the colorectal database, removes non surgical rows such as follow up visit and removes extra columns.

    :returns: cr_sx_all.pickle
    """
    print('pickle_surgeries function is running...')
    df = pd.read_pickle('S:\ERAS\cr_df.pickle')
    #redcap events that should be included (rows)
    redcap_events = ['baseline_arm_1', 'pre_op_visit_dx_1_arm_1', 'surgery_dx_1_arm_1','neo_adjuvant_treat_arm_1','post_op_complicati_arm_1', 'baseline_2_arm_1','pre_op_visit_dx_2_arm_1', 'post_op_complicati_arm_1b','neo_adjuvant_treat_arm_1b']

    df_surgery_all = df[['patient_id','sx_admission_date_a','sx_urgency_a','surgery_mode_a',
    'prim_sx_rectalca_a___7','prim_sx_rectalca_a___8','prim_sx_rectalca_a___9','prim_sx_rectalca_a___10','prim_sx_rectalca_a___25','prim_sx_rectalca_a___11','prim_sx_rectalca_a___31','prim_sx_rectalca_a___30','prim_sx_rectalca_a___13','prim_sx_rectalca_a___14','prim_sx_rectalca_a___15','prim_sx_rectalca_a___27','prim_sx_rectalca_a___24','prim_sx_rectalca_a___16','prim_sx_rectalca_a___17','prim_sx_rectalca_a___18','prim_sx_rectalca_a___19','prim_sx_rectalca_a___28','prim_sx_rectalca_a___29','prim_sx_rectalca_a___20','prim_sx_rectalca_a___21','prim_sx_rectalca_a___22','prim_sx_rectalca_a___23','prim_sx_other_rectalca_a',
    'prim_sx_rectalpolyp_a___7','prim_sx_rectalpolyp_a___8','prim_sx_rectalpolyp_a___9','prim_sx_rectalpolyp_a___10','prim_sx_rectalpolyp_a___25','prim_sx_rectalpolyp_a___11','prim_sx_rectalpolyp_a___12','prim_sx_rectalpolyp_a___30','prim_sx_rectalpolyp_a___29','prim_sx_rectalpolyp_a___13','prim_sx_rectalpolyp_a___26','prim_sx_rectalpolyp_a___14','prim_sx_rectalpolyp_a___15','prim_sx_rectalpolyp_a___16','prim_sx_rectalpolyp_a___24','prim_sx_rectalpolyp_a___27','prim_sx_rectalpolyp_a___17','prim_sx_rectalpolyp_a___18','prim_sx_rectalpolyp_a___19','prim_sx_rectalpolyp_a___28','prim_sx_rectalpolyp_a___20','prim_sx_rectalpolyp_a___21','prim_sx_rectalpolyp_a___22','prim_sx_rectalpolyp_a___23','prim_sx_other_rectlpolyp_a',
    'prim_sx_colonca_a___7','prim_sx_colonca_a___8','prim_sx_colonca_a___9','prim_sx_colonca_a___10','prim_sx_colonca_a___11','prim_sx_colonca_a___12','prim_sx_colonca_a___32','prim_sx_colonca_a___13','prim_sx_colonca_a___14','prim_sx_colonca_a___15','prim_sx_colonca_a___16','prim_sx_colonca_a___35','prim_sx_colonca_a___36','prim_sx_colonca_a___34','prim_sx_colonca_a___29','prim_sx_colonca_a___28','prim_sx_colonca_a___17','prim_sx_colonca_a___18','prim_sx_colonca_a___19','prim_sx_colonca_a___27','prim_sx_colonca_a___20','prim_sx_colonca_a___30','prim_sx_colonca_a___21','prim_sx_colonca_a___22','prim_sx_colonca_a___31','prim_sx_colonca_a___23','prim_sx_colonca_a___24','prim_sx_colonca_a___25','prim_sx_colonca_a___26','prim_sx_other_colonca_a',
    'prim_sx_colonpolyp_a___7','prim_sx_colonpolyp_a___8','prim_sx_colonpolyp_a___9','prim_sx_colonpolyp_a___10','prim_sx_colonpolyp_a___11','prim_sx_colonpolyp_a___12','prim_sx_colonpolyp_a___32','prim_sx_colonpolyp_a___13','prim_sx_colonpolyp_a___14','prim_sx_colonpolyp_a___15','prim_sx_colonpolyp_a___16','prim_sx_colonpolyp_a___33','prim_sx_colonpolyp_a___34','prim_sx_colonpolyp_a___35','prim_sx_colonpolyp_a___29','prim_sx_colonpolyp_a___28','prim_sx_colonpolyp_a___17','prim_sx_colonpolyp_a___18','prim_sx_colonpolyp_a___19','prim_sx_colonpolyp_a___20','prim_sx_colonpolyp_a___30','prim_sx_colonpolyp_a___21','prim_sx_colonpolyp_a___27','prim_sx_colonpolyp_a___22','prim_sx_colonpolyp_a___31','prim_sx_colonpolyp_a___23','prim_sx_colonpolyp_a___24','prim_sx_colonpolyp_a___25','prim_sx_colonpolyp_a___26','prim_sx_other_colonpolyp_a',
    'prim_sx_bencolon_a___1','prim_sx_bencolon_a___2','prim_sx_bencolon_a___3','prim_sx_bencolon_a___4','prim_sx_bencolon_a___5','prim_sx_bencolon_a___6','prim_sx_bencolon_a___7','prim_sx_bencolon_a___8','prim_sx_bencolon_a___9','prim_sx_bencolon_a___10','prim_sx_bencolon_a___11','prim_sx_bencolon_a___12','prim_sx_bencolon_a___13','prim_sx_bencolon_a___14','prim_sx_bencolon_a___15','prim_sx_bencolon_a___16','prim_sx_bencolon_a___25','prim_sx_bencolon_a___17','prim_sx_bencolon_a___18','prim_sx_bencolon_a___24','prim_sx_bencolon_a___19','prim_sx_bencolon_a___26','prim_sx_bencolon_a___27','prim_sx_bencolon_a___20','prim_sx_bencolon_a___21','prim_sx_bencolon_a___22','prim_sx_bencolon_a___23','prim_sx_other_bencolon_a',
    'sx_rectopexy_a',
    'prim_sx_uc_a___1','prim_sx_uc_a___2','prim_sx_uc_a___31','prim_sx_uc_a___3','prim_sx_uc_a___4','prim_sx_uc_a___5','prim_sx_uc_a___24','prim_sx_uc_a___25','prim_sx_uc_a___6','prim_sx_uc_a___7','prim_sx_uc_a___8','prim_sx_uc_a___29','prim_sx_uc_a___9','prim_sx_uc_a___10','prim_sx_uc_a___11','prim_sx_uc_a___22','prim_sx_uc_a___23','prim_sx_uc_a___12','prim_sx_uc_a___13','prim_sx_uc_a___14','prim_sx_uc_a___15','prim_sx_uc_a___26','prim_sx_uc_a___16','prim_sx_uc_a___21','prim_sx_uc_a___27','prim_sx_uc_a___28','prim_sx_uc_a___17','prim_sx_uc_a___18','prim_sx_uc_a___19','prim_sx_uc_a___20','prim_sx_other_uc_a',
    'prim_sx_ic_a___1','prim_sx_ic_a___2','prim_sx_ic_a___30','prim_sx_ic_a___31','prim_sx_ic_a___3','prim_sx_ic_a___4','prim_sx_ic_a___24','prim_sx_ic_a___25','prim_sx_ic_a___5','prim_sx_ic_a___6','prim_sx_ic_a___7','prim_sx_ic_a___8','prim_sx_ic_a___29','prim_sx_ic_a___9','prim_sx_ic_a___10','prim_sx_ic_a___11','prim_sx_ic_a___22','prim_sx_ic_a___23','prim_sx_ic_a___12','prim_sx_ic_a___13','prim_sx_ic_a___14','prim_sx_ic_a___15','prim_sx_ic_a___26','prim_sx_ic_a___16','prim_sx_ic_a___21','prim_sx_ic_a___27','prim_sx_ic_a___28','prim_sx_ic_a___17','prim_sx_ic_a___18','prim_sx_ic_a___19','prim_sx_ic_a___20','prim_sx_other_ic_a',
    'prim_sx_cd_a___1','prim_sx_cd_a___2','prim_sx_cd_a___30','prim_sx_cd_a___31','prim_sx_cd_a___3','prim_sx_cd_a___4','prim_sx_cd_a___24','prim_sx_cd_a___25','prim_sx_cd_a___5','prim_sx_cd_a___6','prim_sx_cd_a___7','prim_sx_cd_a___8','prim_sx_cd_a___29','prim_sx_cd_a___9','prim_sx_cd_a___10','prim_sx_cd_a___11','prim_sx_cd_a___22','prim_sx_cd_a___23','prim_sx_cd_a___26','prim_sx_cd_a___12','prim_sx_cd_a___13','prim_sx_cd_a___14','prim_sx_cd_a___15','prim_sx_cd_a___16','prim_sx_cd_a___21','prim_sx_cd_a___27','prim_sx_cd_a___28','prim_sx_cd_a___17','prim_sx_cd_a___18','prim_sx_cd_a___19','prim_sx_cd_a___20','prim_sx_other_cd_a',
    'sx_multivisc_rxn_a','sx_anastomosis_a','sx_anastamosis_ibd_a','sx_temp_diversion_a','secondary_sx_a___17','secondary_sx_a___18','secondary_sx_a___19','secondary_sx_a___20','secondary_sx_a___21','secondary_sx_a___22','secondary_sx_a___23','secondary_sx_a___24','secondary_sx_a___25','secondary_sx_a___26','secondary_sx_a___27','secondary_sx_a___28','secondary_sx_a___29','secondary_sx_a___30','other_secondary_sx_a','sx_comb_service_a___16','sx_comb_service_a___17','sx_comb_service_a___18','sx_comb_service_a___19','sx_comb_service_a___20','sx_comb_service_a___21','sx_ebl_a','sx_length_a']]

    df_surgery_all.to_pickle('S:\ERAS\cr_sx_all.pickle')

def pickle_comp():
    """
    Takes in main cr_df pickle and selects the complications columns. Only looks at group/event a. Also selects based on redcap event names. There will be one patient per row. The function will only loop through patients who had surgery. This was determined by those who had data in redcap event 'surgery_dx_1_arm_1'

    :returns: cr_df_comp_final.pickle
    """
    print('pickle_comp function is running...')
    df = pd.read_pickle('S:\ERAS\cr_df.pickle') #reads in full data from load_and_pickle function (all data)

    #redcap events that should be included (rows)
    redcap_events = ['baseline_arm_1', 'pre_op_visit_dx_1_arm_1', 'surgery_dx_1_arm_1','neo_adjuvant_treat_arm_1','post_op_complicati_arm_1', 'baseline_2_arm_1','pre_op_visit_dx_2_arm_1', 'post_op_complicati_arm_1b','neo_adjuvant_treat_arm_1b']
    
    #all complications columns (a)
    df_comp = df[['redcap_event_name','patient_id','post_op_compl_a_dx','po_complication_a___1','po_complication_a___2','po_complication_a___3','po_complication_a___16','po_complication_a___4','po_complication_a___5','po_complication_a___6','po_complication_a___7','po_complication_a___8','po_complication_a___9','po_complication_a___17','po_complication_a___15','po_complication_a___13','po_complication_a___10','po_complication_a___14','po_complication_a___11','po_complication_a___12','po_complication_sx_a','po_leak_repair_a','po_sx_bleeding_repair','po_sx_bowel_obstrctn_a','po_sx_dpwoundinfection_a','po_sx_entericfistula_a','po_sx_dehiscence_a','po_sx_hemorrhage_a','po_sx_hernia_a','po_sx_ischemia_a','po_sx_intraab_infection_a','po_sx_intraab_bleed_a','po_sx_readmission_a','po_sx_superficialwound_a','po_sx_urinary_dysfnctn_a','comments_po_compl_a','po_med_complication_a___1','po_med_complication_a___2','po_med_complication_a___3','po_med_complication_a___4','po_med_complication_a___6','po_med_complication_a___5','po_compl_baselinecr_a','po_compl_elevatedcr_a','po_compl_arf_a','po_medcompl_afib_a','po_other_medcompl_a','po_compl_death_a','po_compl_dod_a','po_compl_cod_a','post_op_complications_a_complete']]
    
    df_comp = df_comp[df_comp.redcap_event_name.isin(redcap_events)] #removes rows that are not needed 
    df_comp = df_comp.drop(['redcap_event_name'],axis=1) #drops redcap event name
    df_comp_final = []

    redcap_events = ['surgery_dx_1_arm_1'] #redcap events that should be included (rows)
    pt_with_sx_list = df.patient_id[df.redcap_event_name.isin(redcap_events)].unique().tolist()

    percentage=0 #keeps track of runtime
    # pt_list = df_comp.patient_id.unique()
    pt_list = pt_with_sx_list
    num_of_pts = len(pt_list)

    #loops through each patient
    for cnt, patient in enumerate(pt_list):

        #prints progress
        percentage = project_modules.running_fxn(20,percentage,cnt,num_of_pts)

        df_pt = df_comp[df_comp.patient_id==patient] #pt specific df
        df_pt_cleaned = df_pt.ix[:,df_pt.columns != 'patient_id'].dropna(how='all') #drops rows that have all nan values

        if df_pt_cleaned.shape[0] == 0:
            df_pt_cleaned.loc[len(df_pt_cleaned)] = np.nan #adds a row of NaNs
            df_pt_cleaned['patient_id']=patient #adds back patient_id and sets it equal to the patient id
            df_comp_final.append(df_pt_cleaned) #appends df_comp_final
        elif df_pt_cleaned.shape[0] ==1:
            df_pt_cleaned['patient_id']=patient #adds back patient_id and sets it equal to the patient id
            df_comp_final.append(df_pt_cleaned) #appends df_comp_final
        else:
            print('row:{} pt:{}'.format(df_pt_cleaned.shape[0],patient)) #if more than 2 rows for a pt

    df_comp_final = pd.concat(df_comp_final) #don't put in for loop as it will lead to quadratic copying
    print(df_comp_final.shape)
    
    pd.to_pickle(df_comp_final, 'S:\ERAS\cr_comp.pickle')

def max_complication():
    """
    Takes in cr_comp pickle and complications_dictionary_table pickle to calculate the max complication for each patient

    :returns: cr_comp_score.pickle
    """
    print('max_complication function is running...')
    df_sx_comp = pd.read_pickle('S:\ERAS\cr_comp.pickle')
    # print(df_sx_comp.shape)
    # del df_sx_comp['num_surgeries']
    df_comp_dict = pd.read_pickle('S:\ERAS\complications_dictionary_table.pickle')
    max_result_list = []
    score = 0
    patient = 0
    percentage=0 #keeps track of runtime
    num_of_pts = df_sx_comp.shape[0]

    for cnt,pt in enumerate(df_sx_comp.iterrows()):
        # cnt+=1
        # print(cnt)
        percentage = project_modules.running_fxn(20,percentage,cnt,num_of_pts)
        max_comp_score = [0] #max score list for each pt
        pt_comp_list = pt[1][(pt[1].notnull()) & (pt[1]!=0)] #returns list of vales that are not nan nor 0 (1 or string)
        patient = pt[0]

        for comp in range(0,len(pt_comp_list)):
      
      #checks to see if the value in the CR database is a string rather than a number
            if isinstance(pt_comp_list[comp],str):
                score = df_comp_dict[df_comp_dict.comp_name==pt_comp_list[comp]].score.values[0]                
            
            #otherwise it is a number
            else:    
                #get errors with the sx columns that were still included. All would be scores of 3 so not needed
                try:
                    score = df_comp_dict[df_comp_dict.comp_name==pt_comp_list.index[comp]].score.values[0]
                except IndexError:
                    pass
                    #print('patient: {} comp: {}'.format(pt[0],pt_comp_list.index[comp]))
            max_comp_score.append(score) #adds score to pt list
        max_result_list.append(max(max_comp_score)) #adds max score for the pt to a master list

    df_sx_comp['comp_score']=max_result_list
    df_comp_score = pd.DataFrame({'patient_id':df_sx_comp.patient_id,'comp_score':df_sx_comp.comp_score})
    pd.to_pickle(df_comp_score,'S:\ERAS\cr_comp_score.pickle')

def create_sx_dict():
    """
    Takes data from a list of surgery names (sx_list_input.xlsx) and addes the appropriate numbers to form all unique headers.   
        -iterates over the input file to create all unique column names
        -then iterates over all procedures in dictionary to find a match
        -values in this are set up in specific order to make sure incorrect matches do not occur
        -outputs a file with column name, surgical score, description, unique name, unique code/number    

    :returns: sx_list_dict_comp xlsx and pickle file. These files will act as a mapping dictionary with header names, surgery group (score), description and associated code to unify list.
    """
    df = pd.read_excel('S:\ERAS\sx_list_input.xlsx')
    main_output = [] #will be column name in CR database
    description_output = [] #description of column
    score = []
    unique_list = [] #unique names
    unique_code = [] #unique codes for each procedure

    #this is a workaround where the surgery group assingment file is read, transposed, written to another excel and then read back in to a pandas dataframe. There were issues with the formating if this was not done. There is probably a fix, but not a big deal at this point.
    temp_df = pd.read_excel('S:\ERAS\surgery_group_assigment.xlsx')
    temp_df = temp_df.T
    temp_df.to_excel('S:\ERAS\surgery_group_assigment_transposed.xlsx',index=False,header=False)
    df_unique = pd.read_excel('S:\ERAS\surgery_group_assigment_transposed.xlsx')
    
    procedure_dict = df_unique.to_dict()  

    #iterate over all rows
    for row in df.iterrows():
        input_name = row[1].values[0] #gets main name
        text_names = row[1].values[1] #gets long string with values separated by "|", except for a few that have no string
        
        #for cells that have a strings separated by "|"
        try:
            text_list = text_names.split(' | ')
            for item in text_list:
                match = False
                for procedure in procedure_dict:      
                    find_procedure = re.search(procedure,item)
                    if find_procedure is not None:
                        unique_list.append(procedure_dict[procedure][0])
                        unique_code.append(procedure_dict[procedure][1])
                        score.append(procedure_dict[procedure][2])
                        # print(procedure_dict[procedure][2])
                        match = True
                        break #if match no need to look further
                #checks to make sure there was a match
                if not match:
                    unique_list.append('None')
                    unique_code.append(-1)
                    score.append(-1)
                regex = re.search(r'(\w+).*',item)
                number = regex.group(1) #number value from string
                procedure = regex.group(0) #whole string
                main_output.append('{}___{}'.format(input_name,number)) #creates the unique name for each value in second column separted by "|"
                description_output.append(procedure)
        
        #for cells that do not have a string in the second column
        except:
            item = input_name
            match = False
            for procedure in procedure_dict:      
                find_procedure = re.search(procedure,item)
                if find_procedure is not None:
                    unique_list.append(procedure_dict[procedure][0])
                    unique_code.append(procedure_dict[procedure][1])
                    score.append(procedure_dict[procedure][2])
                    match = True
                    break #if match no neeed to look further

            #checks to make sure there was a match
            if not match:
                unique_list.append('None')
                unique_code.append(-1)
                score.append(-1)
            regex = re.search(r'(\w+).*',item)
            number = regex.group(1)
            procedure = regex.group(0)
            main_output.append(input_name)
            description_output.append(procedure)
                    

    df_out = pd.DataFrame(main_output,columns=['name'])
    df_out['score'] = score
    df_out['description'] = description_output
    df_out['unique'] = unique_list
    df_out['code'] = unique_code
    df_out.to_excel('S:\ERAS\sx_list_dict_comp.xlsx',sheet_name='Sheet1')

    pd.to_pickle(df_out,'S:\ERAS\sx_list_dict_comp.pickle')

def organize_sx():
    """
    Looks at cr_sx_all and sx_list_dict_comp pickles to determine surgical score. It will onehotencode by group (4 groups)

    :returns: df_sx_score.pickle
    """
    print('organize_sx function is running...')
    df = pd.read_pickle('S:\ERAS\cr_sx_all.pickle')
    df_sx_dict_comp = pd.read_pickle('S:\\ERAS\sx_list_dict_comp.pickle')

    #removes emergent cases (2), elective are (1).
    df = df[df.sx_urgency_a==1]

    pt_list = list(df.patient_id.unique()) #creates list of patients for more effecient looping
    num_of_pts = len(pt_list)
   
    df.drop(['sx_urgency_a'],axis=1,inplace=True) #removes non surgical columns
    
    percentage=0 #keeps track of runtime
    group_dict = {1:[],2:[],3:[],4:[]}

    for cnt,patient in enumerate(pt_list):
        percentage = project_modules.running_fxn(20,percentage,cnt,num_of_pts)
        df_pt = df[df.patient_id==patient] #pt specific df
        df_pt = df_pt.ix[:,df_pt.columns!='patient_id'].dropna(how='all') #drops rows that have all nan values
        df_pt = df_pt.replace(0,np.NaN)
        sx_list = df_pt.columns[pd.notnull(df_pt).sum()>0].tolist()

        # if the patient did not have any surgeries skip
        if df_pt.shape[0]==0:
            pass

        # if only one surgery row
        elif df_pt.shape[0]==1:

            #loops through each surgery
            group_cnt= {1:0,2:0,3:0,4:0}
            for cnt, sx in enumerate(df_pt.items()):
                # print(cnt)
                
                if sx[1].values[0]==1: #if the sx column has a value of 1 meaning it happened
                    try:
                        score = df_sx_dict_comp.score[df_sx_dict_comp.name==sx[0]].values[0] #surgical score which will correlate to groupby
                    except:
                        score = -1

                    if score == 1 and group_cnt[1] != 1:
                        group_dict[1].append(1)
                        group_cnt[1] = 1
                    elif score == 2 and group_cnt[2] != 1:
                        group_dict[2].append(1)
                        group_cnt[2] = 1
                    elif score == 3 and group_cnt[3] != 1:
                        group_dict[3].append(1)
                        group_cnt[3] = 1
                    elif score == 4 and group_cnt[4] != 1:
                        group_dict[4].append(1)
                        group_cnt[4] = 1
                    elif score == -1:
                        pass

            for i in group_cnt:
            	if group_cnt[i] == 0:
            		group_dict[i].append(0)

        #will just take the first operation for now
        elif df_pt.shape[0]==2:
            group_cnt= {1:0,2:0,3:0,4:0}

            df_pt = df_pt.iloc[[0]]
            for cnt, sx in enumerate(df_pt.items()):
                if sx[1].values[0]==1: #if the sx column has a value of 1 meaning it happened
                    #comb_service are not in dictionary and will throw an error
                   
                    if sx[0] in df_sx_dict_comp.name:
                        score = df_sx_dict_comp.score[df_sx_dict_comp.name==sx[0]].values[0] #surgical score which will correlate to groupby

                    if score == 1 and group_cnt[1] != 1:
                        group_dict[1].append(1)
                        group_cnt[1] = 1
                    elif score == 2 and group_cnt[2] != 1:
                        group_dict[2].append(1)
                        group_cnt[2] = 1
                    elif score == 3 and group_cnt[3] != 1:
                        group_dict[3].append(1)
                        group_cnt[3] = 1
                    elif score == 4 and group_cnt[4] != 1:
                        group_dict[4].append(1)
                        group_cnt[4] = 1
                    elif score == -1:
                    	pass

            for i in group_cnt:
                if group_cnt[i] == 0:
                    group_dict[i].append(0)

        else:
            print('More than 2 rows')

    df_sx_score = pd.DataFrame({'patient_id':pt_list,'group_1':group_dict[1],'group_2':group_dict[2],'group_3':group_dict[3],'group_4':group_dict[4]})
    pd.to_pickle(df_sx_score,'S:\ERAS\df_sx_score.pickle')
    df_sx_score.to_excel('S:/ERAS/df_sx_score.xlsx')

def pickle_demographics():
    """
    Reads cr_df pickle and selects the demographics columns.

    :returns: df_demogrpahics.pickle
    """
    df = pd.read_pickle('S:\ERAS\cr_df.pickle')
    df_demo = df[['patient_id','redcap_event_name','age','sex','race','ethnicity','bmi','primary_dx','other_dx_','second_dx','other_second_dx','pt_hx_statusdiv___1','pt_hx_statusdiv___2','pt_hx_statusdiv___3','no_compl_attacks','no_divattacks_hospital','no_total_attacks','no_ab_sx','prior_ab_sx___0','prior_ab_sx___1','prior_ab_sx___2','prior_ab_sx___3','prior_ab_sx___4','prior_ab_sx___5','prior_ab_sx___6','prior_ab_sx___7','prior_ab_sx___8','prior_ab_sx___9','prior_ab_sx___10','prior_ab_sx___11','prior_ab_sx___12','prior_ab_sx___13','prior_ab_sx___14','prior_ab_sx___15','prior_ab_sx___16','prior_ab_sx___17','prior_ab_sx___18','prior_ab_sx___19','prior_ab_sx_other','med_condition___1','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','med_condition___10','med_condition___11','med_condition___12','med_condition___13','current_medtreatment___14','current_medtreatment___15','current_medtreatment___16','current_medtreatment___17','current_medtreatment___18','current_medtreatment___19','current_medtreatment___20','current_medtreatment___21','current_medtreatment___22','current_medtreatment___23','asa_class','ho_smoking','cea_value','wbc_value','hgb_value','plt_value','bun_value','creatinine_value','albumin_value','alp_value','glucose_value','hba1c_value','prealbumin_value','crp_value']]

    pd.to_pickle(df_demo,'S:\ERAS\df_demographics.pickle')

def try_baseline(df,demographic):
    """
    Takes a patient df and the demographic and logically goes through to get values regardless of how many rows there are.

    :returns: demographic value
    """
    if df.shape[0]==0:
        print('Error: no demographics') #error catch
    elif df.shape[0]==1:
        result = df[demographic].values[0]
    else:
        try:
            result = df[demographic][df[demographic].notnull()].values[0]
        except IndexError:
            result = df[demographic].values[0]
    return result

def organize_demographics():
    """
    Takes in demographics pickle and sx score pickle. It goes through and condenses demographics, medical conditions, medical treatment, asa, smoking, labs, etc.

    :returns: df_demographics_out.pickle
    """
    print('organize_demographics function is running...')
    df_demo = pd.read_pickle('S:\ERAS\df_demographics.pickle') #reads in demographics df
    df_sx_score = pd.read_pickle('S:\ERAS\df_sx_score.pickle') #reads in sx score df for relevant pts
    df_demo = pd.merge(df_sx_score,df_demo,on="patient_id",how='left') #merges on sx score df

    #redcap rows/groups
    redcap_events = ['baseline_arm_1','pre_op_visit_dx_1_arm_1','baseline_2_arm_1','pre_op_visit_dx_2_arm_1']
    redcap_baseline = ['baseline_arm_1','baseline_2_arm_1']
    redcap_preop = ['pre_op_visit_dx_1_arm_1','pre_op_visit_dx_2_arm_1']

    #column groups for medical conditions and treatment
    med_cond = ['med_condition___1','med_condition___2','med_condition___3','med_condition___4','med_condition___5','med_condition___6','med_condition___7','med_condition___8','med_condition___9','med_condition___10','med_condition___11','med_condition___12','med_condition___13']
    med_tx = ['current_medtreatment___14','current_medtreatment___15','current_medtreatment___16','current_medtreatment___17','current_medtreatment___18','current_medtreatment___19','current_medtreatment___20','current_medtreatment___21','current_medtreatment___22','current_medtreatment___23']
    asa_smoke = ['asa_class','ho_smoking']
    labs = ['cea_value','wbc_value','hgb_value','plt_value','bun_value','creatinine_value','albumin_value','alp_value','glucose_value','hba1c_value','prealbumin_value','crp_value']

    df_demo = df_demo[df_demo.redcap_event_name.isin(redcap_events)] #removes rows that are not needed defined by redcap events list

    df_med_cond_0 = pd.DataFrame({'med_condition___9': [0], 'med_condition___1': [0], 'med_condition___10': [0], 'med_condition___4': [0], 'med_condition___8': [0], 'med_condition___6': [0], 'med_condition___2': [0], 'med_condition___12': [0], 'med_condition___13': [0], 'med_condition___11': [0], 'med_condition___7': [0], 'med_condition___5': [0], 'med_condition___3': [0]})

    #initiates baseline
    age_list = []
    sex_list = []
    race_list = []
    ethnicity_list = []
    bmi_list = []
    primary_dx_list = []
    second_dx_list = []
    no_total_attacks_list = []
    no_ab_sx_list = []
    med_cond_list = []
    med_tx_list = []

    #initiates med conditions
    med_condition___1_list = []
    med_condition___2_list = []
    med_condition___3_list = []
    med_condition___4_list = []
    med_condition___5_list = []
    med_condition___6_list = []
    med_condition___7_list = []
    med_condition___8_list = []
    med_condition___9_list = []
    med_condition___10_list = []
    med_condition___11_list = []
    med_condition___12_list = []
    med_condition___13_list = []

    #initiates med treatments
    current_medtreatment___14_list = []
    current_medtreatment___15_list = []
    current_medtreatment___16_list = []
    current_medtreatment___17_list = []
    current_medtreatment___18_list = []
    current_medtreatment___19_list = []
    current_medtreatment___20_list = []
    current_medtreatment___21_list = []
    current_medtreatment___22_list = []
    current_medtreatment___23_list = []

    #initiates asa/smoking
    asa_class_list = []
    ho_smoking_list = []

    #initiates labs
    cea_value_list = []
    wbc_value_list = []
    hgb_value_list = []
    plt_value_list = []
    bun_value_list = []
    creatinine_value_list = []
    albumin_value_list = []
    alp_value_list = []
    glucose_value_list = []
    hba1c_value_list = []
    prealbumin_value_list = []
    crp_value_list = []

    pt_list = list(df_sx_score.patient_id)
    num_of_pts = len(pt_list)

    percentage=0 #keeps track of runtime

    #loops through all patients in list
    for cnt, patient in enumerate(pt_list):

        percentage = project_modules.running_fxn(20,percentage,cnt,num_of_pts)

        #dateframes
        df_pt = df_demo[df_demo.patient_id==patient] #pt df
        df_pt_baseline = df_pt[df_pt.redcap_event_name.isin(redcap_baseline)] #pt baseline df
        df_pt_preop = df_pt[df_pt.redcap_event_name.isin(redcap_preop)] #pt preop eval df
        df_pt_med_cond = df_pt_baseline[med_cond] #med condition df
        df_pt_med_tx = df_pt_baseline[med_tx] #med treatement df
        df_pt_asa_smoke = df_pt_baseline[asa_smoke] #asa and smoke df (from list object asa_smoke)
        df_pt_labs = df_pt_preop[labs]

        #baseline characteristics
        age = try_baseline(df_pt_baseline,'age')
        sex = try_baseline(df_pt_baseline,'sex')
        race = try_baseline(df_pt_baseline,'race')
        ethnicity = try_baseline(df_pt_baseline,'ethnicity')
        bmi = try_baseline(df_pt_baseline,'bmi')
        primary_dx =try_baseline(df_pt_baseline,'primary_dx')
        second_dx = try_baseline(df_pt_baseline,'second_dx')
        no_total_attacks = try_baseline(df_pt_baseline,'no_total_attacks')
        no_ab_sx = try_baseline(df_pt_baseline,'no_ab_sx')

        #adds to each list
        age_list.append(age)
        sex_list.append(sex)
        race_list.append(race)
        ethnicity_list.append(ethnicity)
        bmi_list.append(bmi)
        primary_dx_list.append(primary_dx)
        second_dx_list.append(primary_dx)
        no_total_attacks_list.append(no_total_attacks)
        no_ab_sx_list.append(no_ab_sx)

        #medical conditions
        max_pt_med_cond = df_pt_med_cond.max() #gets max for medical conditions (values should only be nan, 0, and 1)

        #adds to each list
        med_condition___1_list.append(max_pt_med_cond[0])
        med_condition___2_list.append(max_pt_med_cond[1])
        med_condition___3_list.append(max_pt_med_cond[2])
        med_condition___4_list.append(max_pt_med_cond[3])
        med_condition___5_list.append(max_pt_med_cond[4])
        med_condition___6_list.append(max_pt_med_cond[5])
        med_condition___7_list.append(max_pt_med_cond[6])
        med_condition___8_list.append(max_pt_med_cond[7])
        med_condition___9_list.append(max_pt_med_cond[8])
        med_condition___10_list.append(max_pt_med_cond[9])
        med_condition___11_list.append(max_pt_med_cond[10])
        med_condition___12_list.append(max_pt_med_cond[11])
        med_condition___13_list.append(max_pt_med_cond[12])

        #medical treatment
        max_pt_med_tx = df_pt_med_tx.max() #gets max for medical treatment (values should only be nan, 0, and 1)

        #adds to each list
        current_medtreatment___14_list.append(max_pt_med_tx[0])
        current_medtreatment___15_list.append(max_pt_med_tx[1])
        current_medtreatment___16_list.append(max_pt_med_tx[2])
        current_medtreatment___17_list.append(max_pt_med_tx[3])
        current_medtreatment___18_list.append(max_pt_med_tx[4])
        current_medtreatment___19_list.append(max_pt_med_tx[5])
        current_medtreatment___20_list.append(max_pt_med_tx[6])
        current_medtreatment___21_list.append(max_pt_med_tx[7])
        current_medtreatment___22_list.append(max_pt_med_tx[8])
        current_medtreatment___23_list.append(max_pt_med_tx[9])

        #asa and smoking
        max_pt_asa_smoke = df_pt_asa_smoke.max()

        #adds to each list
        asa_class_list.append(max_pt_asa_smoke[0])
        ho_smoking_list.append(max_pt_asa_smoke[1])

        #labs
        median_labs = df_pt_labs.median()

        #adds to each list
        cea_value_list.append(median_labs[0])
        wbc_value_list.append(median_labs[1])
        hgb_value_list.append(median_labs[2])
        plt_value_list.append(median_labs[3])
        bun_value_list.append(median_labs[4])
        creatinine_value_list.append(median_labs[5])
        albumin_value_list.append(median_labs[6])
        alp_value_list.append(median_labs[7])
        glucose_value_list.append(median_labs[8])
        hba1c_value_list.append(median_labs[9])
        prealbumin_value_list.append(median_labs[10])
        crp_value_list.append(median_labs[11])

        # if round(cnt/num_of_pts*100) != percentage:
        #     percentage = round(cnt/num_of_pts*100)
        #     if percentage in range(0,101,5):
        #         print('{}% complete'.format(percentage))

    df_output = pd.DataFrame({'patient_id':pt_list,'age':age_list,'sex':sex_list,'race':race_list,'ethnicity':ethnicity_list,'bmi':bmi_list,'primary_dx':primary_dx_list,'second_dx':second_dx_list,'no_total_attacks':no_total_attacks_list,'no_ab_sx':no_ab_sx_list,'med_condition___9': med_condition___9_list, 'med_condition___1': med_condition___1_list, 'med_condition___10': med_condition___10_list, 'med_condition___4': med_condition___4_list, 'med_condition___8': med_condition___8_list, 'med_condition___6': med_condition___6_list, 'med_condition___2': med_condition___2_list, 'med_condition___12': med_condition___12_list, 'med_condition___13': med_condition___13_list, 'med_condition___11': med_condition___11_list, 'med_condition___7': med_condition___7_list, 'med_condition___5': med_condition___5_list, 'med_condition___3': med_condition___3_list,'currenct_medtreatment___14':current_medtreatment___14_list,'currenct_medtreatment___15':current_medtreatment___15_list,'currenct_medtreatment___16':current_medtreatment___16_list,'currenct_medtreatment___17':current_medtreatment___17_list,'currenct_medtreatment___18':current_medtreatment___18_list,'currenct_medtreatment___19':current_medtreatment___19_list,'currenct_medtreatment___20':current_medtreatment___20_list,'currenct_medtreatment___21':current_medtreatment___21_list,'currenct_medtreatment___22':current_medtreatment___22_list,'currenct_medtreatment___23':current_medtreatment___23_list,'asa_class':asa_class_list,'ho_smoking':ho_smoking_list,'cea_value':cea_value_list,'wbc_value':wbc_value_list,'hgb_value':hgb_value_list,'plt_value':plt_value_list,'bun_value':bun_value_list,'creatinine_value':creatinine_value_list,'albumin_value':albumin_value_list,'alp_value':alp_value_list,'glucose_value':glucose_value_list,'hba1c_value':hba1c_value_list,'prealbumin_value':prealbumin_value_list,'crp_value':crp_value_list})

    df_output.no_total_attacks.fillna(0,inplace=True)
    df_output.no_ab_sx.fillna(0,inplace=True)

    pd.to_pickle(df_output,'S:\ERAS\df_demographics_out.pickle')

    """
    #will print out numbers for nans
    demo_list = ['age','sex','race','ethnicity','bmi','primary_dx','second_dx','no_total_attacks','no_ab_sx']
    for i in demo_list:
        nans = df_output[df_output[i].isnull()].shape[0]
        print('Demo:{} NaNs:{}'.format(i,nans))
    """    

def readmit_los():
    """
    Reads in cr_df.pickle and df_sx_score.pickle. It will grab los, readmit, ebl,surgery length, diversion status, surgeon list, surgery diagnosis, surgery facility, surgery mode.

    :returns: los_readmit.pickle
    """
    print('readmit_los function is running...')
    df = pd.read_pickle('S:\ERAS\cr_df.pickle')
    df_sx_score = pd.read_pickle('S:\ERAS\df_sx_score.pickle') #reads in sx score df for relevant pts
    df_sx_score = pd.merge(df_sx_score,df,on="patient_id",how='left') #merges on sx score df
    
    #redcap rows used to construct subsets of dataframes
    redcap_sx_list = ['surgery_dx_1_arm_1']
    redcap_comp_list = ['post_op_complicati_arm_1','post_op_complicati_arm_1b']

    #subsets of main df
    df_sx_arm = df_sx_score[df_sx_score.redcap_event_name.isin(redcap_sx_list)]
    df_comp_arm = df_sx_score[df_sx_score.redcap_event_name.isin(redcap_comp_list)]

    los_list = ['patient_id','sx_po_stay_a']
    readmit_list = ['patient_id','po_sx_readmission_a']
    sx_ebl_list = ['patient_id','sx_ebl_a']
    sx_length_list = ['patient_id','sx_length_a']
    sx_diversion_list = ['patient_id','sx_temp_diversion_a']
    surgeon_list = ['patient_id','surgeon_a___1','surgeon_a___2','surgeon_a___3','surgeon_a___4','surgeon_a___5']
    sx_dx_list = ['patient_id','sx_diagnosis_a']
    sx_fac_list = ['patient_id','sx_facility_a']
    sx_mode_list = ['patient_id','surgery_mode_a']

    surgeon_dict = {'patient_id':[],'surgeon_a___1':[],'surgeon_a___2':[],'surgeon_a___3':[],'surgeon_a___4':[],'surgeon_a___5':[]}
    """
    sx_diagnosis_a - one_hot
    surgeon_a___1
    surgeon_a___2
    surgeon_a___3
    surgeon_a___4
    surgeon_a___5
    sx_facility_a - one_hot
    surgery_mode_a - one_hot
    """

    df_los = df_sx_arm[los_list]
    df_readmit = df_comp_arm[readmit_list]
    df_sx_ebl = df_sx_arm[sx_ebl_list]
    df_sx_length = df_sx_arm[sx_length_list]
    df_diversion = df_sx_arm[sx_diversion_list]
    df_surgeon = df_sx_arm[surgeon_list]

    df_sx_dx = df_sx_arm[sx_dx_list]
    df_sx_fac = df_sx_arm[sx_fac_list]
    df_sx_mode = df_sx_arm[sx_mode_list]

    pt_list = list(df_sx_score.patient_id.unique())

    sx_po_stay_list = []
    po_sx_readmission_list = []
    sx_ebl_list = []
    sx_length_list = []
    sx_diversion_list = []
    # surgeon_list = []
    dx_list = []
    fac_list = []
    mode_list = []

    num_of_pts = len(pt_list)

    percentage = 0
    for cnt, patient in enumerate(pt_list):

        #prints out progress
        percentage = project_modules.running_fxn(20,percentage,cnt,num_of_pts)

        df_pt_los = df_los[df_los.patient_id==patient]
        df_pt_los = df_pt_los[['sx_po_stay_a']]
        df_pt_readmit = df_readmit[df_readmit.patient_id==patient]
        df_pt_readmit = df_pt_readmit[['po_sx_readmission_a']]
        df_pt_ebl = df_sx_ebl[df_sx_ebl.patient_id==patient]
        df_pt_ebl = df_pt_ebl[['sx_ebl_a']]
        df_pt_length = df_sx_length[df_sx_length.patient_id==patient]
        df_pt_length = df_pt_length[['sx_length_a']]
        df_pt_diversion = df_diversion[df_diversion.patient_id==patient]
        df_pt_diversion = df_pt_diversion[['sx_temp_diversion_a']]
        df_pt_surgeon = df_surgeon[df_surgeon.patient_id==patient]
        df_pt_surgeon = df_pt_surgeon[surgeon_list]
        df_pt_sx_dx = df_sx_dx[df_sx_dx.patient_id==patient]
        df_pt_sx_dx = df_pt_sx_dx[sx_dx_list]
        df_pt_sx_fac = df_sx_fac[df_sx_fac.patient_id==patient]
        df_pt_sx_fac = df_pt_sx_fac[sx_fac_list]
        df_pt_sx_mode = df_sx_mode[df_sx_mode.patient_id==patient]
        df_pt_sx_mode = df_pt_sx_mode[sx_mode_list]

        #los
        sx_po_stay_list.append(df_pt_los.max().max())

        #readmit
        if df_pt_readmit.notnull().sum().sum()>0:
            po_sx_readmission_list.append(1)
        else:
            po_sx_readmission_list.append(0)

        #ebl/length
        sx_ebl_list.append(df_pt_ebl.max().max())
        sx_length_list.append(df_pt_length.max().max())
        sx_diversion_list.append(df_pt_diversion.max().max())

        #surgeron
        for item in df_pt_surgeon:
            surgeon_dict[item].append(df_pt_surgeon[item].values[0])

        #diagnosis, facility, surgery mode
        dx_list.append(df_pt_sx_dx.values[0][1])
        fac_list.append(df_pt_sx_fac.values[0][1])
        mode_list.append(df_pt_sx_mode.values[0][1])

    df_output = pd.DataFrame({'patient_id':pt_list,'sx_po_stay':sx_po_stay_list,'po_sx_readmission':po_sx_readmission_list,'sx_ebl':sx_ebl_list,'sx_length':sx_length_list,'sx_diversion':sx_diversion_list,'sx_diagnosis':dx_list,'sx_facility':fac_list,'surgery_mode':mode_list})

    df_surgeon = pd.DataFrame(surgeon_dict)
    df_output = pd.merge(df_output,df_surgeon,how='inner',on='patient_id')
    pd.to_pickle(df_output,'S:\ERAS\los_readmit.pickle')
    return df_output

def combine_all():
    """
    hardcoded to combine all of the dataframes (demographics, surgery score/group, complication, los/readmission)

    :returns: cr_preprocess.pickle
    """
    df_demo = pd.read_pickle('S:\ERAS\df_demographics_out.pickle')
    df_sx = pd.read_pickle('S:\ERAS\df_sx_score.pickle')
    df_comp = pd.read_pickle('S:\ERAS\cr_comp_score.pickle')
    df_los_readmit = pd.read_pickle('S:\ERAS\los_readmit.pickle')
    df_sx_comp = pd.merge(df_sx,df_comp,how='inner',on='patient_id')
    df_sx_comp_demo = pd.merge(df_sx_comp,df_demo,how='inner',on='patient_id')
    df_output = pd.merge(df_sx_comp_demo,df_los_readmit,how='inner',on='patient_id')
    
    #selects only patients who fit into one of the 4 groups
    df_output = df_output[(df_output.group_1!=0)|(df_output.group_2!=0)|(df_output.group_3!=0)|(df_output.group_4!=0)]
    pd.to_pickle(df_output,'S:\ERAS\cr_preprocess.pickle')
    return df_output

def reduce_pt_rows(df):
    """
    Not in main program. Just a tool to look into patient data.

    :return: None
    """
    df_unique = df_pt_med_cond[col].groupby(df_pt_med_cond[col]).unique()
    if df_unique.shape[0]==0:
        print('pt:{} col:{} size:{}'.format(patient,col,df_unique.shape[0]))
    elif df_unique.shape[0]==1:
        pass
    else:
        #pt 70 and 1172 are under this. both had a dx which was not dx on second (angina, htn)
        df_unique = df_pt_med_cond[col].groupby(df_pt_med_cond[col]).sum()
        print('pt:{} col:{} size:{}'.format(patient,col,df_unique.shape[0]))

def main():
    """Runs the entire pipeline."""
    # project_modules.load_and_pickle(path_in='S:/ERAS/',file_in='CR_all.xlsx',file_out='cr_df.pickle',sheetname='CR_all')
    # pickle_comp()
    # project_modules.load_and_pickle(path_in='S:/ERAS/',file_in='complications_dictionary_table.xlsx',sheetname='Sheet1')
    # max_complication()
    # pickle_surgeries()
    create_sx_dict()
    organize_sx()
    pickle_demographics()
    organize_demographics()
    readmit_los()
    combine_all()

if __name__ == '__main__':
	main()