# Enhanced Recovery After Surgery (ERAS) Project

##Questions
- Still have issue with anastomotic resection if not otherwise specified (ignore for now)

## To Do
- fix redundancy with transpose
- finish updating the surgery groups once questions are anwsered
- update processing to change surgeries from ranks to categories

## Surgery Groups
### Group 1
- Diverting Stomas (ileosotmy and colostomy)

### Group 2
- Segmental Colectomies (right, left, sigmoid, etc.)
- Small Bowel Resection (and stricturoplasty)
- Ileostomy Reversal
- Total Abdominal Colectomy (TAC)
- Intestinal Bypass

### Group 3
- LAR
- Colostomy Reversal

### Group 4
- TPC /w or /wo IPAA
- PSG
- APR

### Excluded
- Altemeier/Delorme
- Transanal Procedures
- Hernia Repairs
- LOA
- Omentectomy
- Enterotomy
- Treatment of Intrabominal Bleeding
- Wound Dehiscence Repair
- Wound Exploration
- Exploratory Laparotomy
- Anastomotoic Leak Repair
- Lavage and Drain Placement

## Data Inputs
### Impute missing as value
- primary_dx
- race
- second_dx
- sex
- ethnicity
- ho_smoking
- sx_diagnosis
- sx_facility
- surgery_mode
### Impute missing not at random
- currenct_medtreatment___14
- currenct_medtreatment___15
- currenct_medtreatment___16
- currenct_medtreatment___17
- currenct_medtreatment___18
- currenct_medtreatment___19
- currenct_medtreatment___20
- currenct_medtreatment___21
- currenct_medtreatment___22
- currenct_medtreatment___23
- med_condition___1
- med_condition___10
- med_condition___11
- med_condition___12
- med_condition___13
- med_condition___2
- med_condition___3
- med_condition___4
- med_condition___5
- med_condition___6
- med_condition___7
- med_condition___8
- med_condition___9
- cea_value
- crp_value
- no_ab_sx
- no_total_attacks
- sx_diversion
- surgeon_a___1
- surgeon_a___2
- surgeon_a___3
- surgeon_a___4
- surgeon_a___5
### Impute mean
- age
- albumin_value
- alp_value
- bmi
- bun_value
- creatinine_value
- glucose_value
- hgb_value
- plt_value
- prealbumin_value
- wbc_value
- sx_ebl
- sx_length
### Impute mode
- sx_score
- asa_class

## Data Outputs
- surgical readmission (y/n)
- post-op stay (days) - available, but not included now
- Clavien-Dindo Score (0-5)