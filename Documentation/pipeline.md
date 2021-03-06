#Description of CR_pipeline_final.py

CR_all.xlsx	-> cr_df.pickle -> cr_sx_all.pickle  		-> cr_df_sx_comp.pickle ->\ 
							-> cr_df_comp_final_pickle	->/						-> cr_df_comp_score.pickle
				 						  complications_dictionary_table.pickle ->/
				 			-> cr_sx_all.pickle -> 
sx_list_impute.xlsx -> sx_list_dict_comp.pickle ->
CR_unique_dict.xlsx ->/
				 	   

##running_fxn
-simply prints out percentage complete graphically with "#"

##cr_columns
-takes in all of the data and selects only the columns of intererest (too many to list). It returns the dataframe with only these columns

##load_and_pickle
-loads the data from an excel file called "CR_all"
-selects the appropriate data using cr_columns
-writes to a pickle called "cr_df.pickle"

##pickle_surgeries
-gets data from cr_df.pickle
-selects teh relevant redcap events
-selects teh surgical columns
-writes to a pickle called "cr_sx_all.pickle"

##pickle_comp
-also gets data from cr_df.pickle
-also selects relevant redcap events
-selects only complication columns
-writes complications to "cr_df_comp_final.pickle" and "cr_df_comp.pickle"

##sx_comp
-has inputs of cr_df.pickle, cr_df_comp.pickle, and cr_df_comp_final.pickle
-looks only at surgery arm in redcap events
-writes to "cr_df_sx_comp.pickle"

##pickle_comp_dict
-reads complications_dictionary_table.xlsx which is a file that has the complication name (including free text values) and assigns a score to each one.
-writes it to a pickle file

##max_complication
-loads from cr_df_sx_comp.pickle and complications_dictionary_table.pickle
-uses the the two files to find out the max complication score for each patient
-writes to cr_df_comp_score.pickle

##pickle_surgeries
-loads data from cr_df
-selects only surgery columns and writes to cr_sx_all.pickle

##create_sx_dict
-takes in excel file called sx_list_impute.xlsx and CR_unique_dict.xlsx
-outputs to sx_list_dict_comp.pickle