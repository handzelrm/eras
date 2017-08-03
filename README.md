# Enhanced Recovery After Surgery (ERAS) Project

## Questions
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
#### primary_dx:
- Group 0: rectal cancer, retal polpys, colon cancer, colon polyps, rectal mass, colon mass, recurrent colon cancer with mets, recurrent rectal cancer with mets
- Group 1: Crohns disease, ulcerative colitis
- Group 2: Ischemic Colitis
- Group 3: Diverticulitis
- Group 4: colonic inertia
- Group 5: other
- Group 6: nan
#### race
#### second_dx
#### sex
#### ethnicity
#### ho_smoking
#### sx_diagnosis
#### sx_facility
#### surgery_mode
- Group 0: open, lap converted
- Group 1: hand-assisted
- Group 2: laparoscopic, roboitc, laparscopic/robotic
- Group 3: TA TME
- Group 4: nan
### Impute missing not at random
#### Cardiac
- med_condition___1 (afib)
- med_condition___2 (Angina)
- med_condition___3 (Cardiac Pacemaker)
- med_condition___4 (CHF)
- med_condition___9 (Previous MI)
#### Renal
- med_condion___5 (Chronic Renal Failure)
#### COPD
- med_condion___6 (COPD)
#### Diabetes
- med_condion___7 (Diabetes)
#### Hypertension
- med_condion___8 (Hypertension)
#### Radiation
- med_condion___10 (Radiation)
#### Transplant
- med_condion___13 (Transplant)
#### Medication Treatement
- currenct_medtreatment___14 (Biologic for cancer)
- currenct_medtreatment___15 (Biologic for IBD)
- currenct_medtreatment___16 (Chemo)
- currenct_medtreatment___17 (Coumadin)
- currenct_medtreatment___18 (Heparin)
- currenct_medtreatment___19 (IV Steroids)
- currenct_medtreatment___20 (Oral Steroids)
- currenct_medtreatment___21 (Topical Steroids)
- currenct_medtreatment___22 (Plavix)
- currenct_medtreatment___23 (Radiation)
#### Labs
- cea_value
- crp_value
#### Other
- no_ab_sx (number of abdominal surgeries)
- no_total_attacks (number of diverticular attacks)
- sx_diversion (surgical diversion)
#### Surgeon
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