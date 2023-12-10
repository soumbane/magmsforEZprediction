# Use this to generate subgroups: MR0(MR-: MR=0)/MR1(MR+: MR=1)/SF(SF=1)/SR(SF=0) for validation patients for all nodes in left hemisphere
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import loadmat, savemat
import openpyxl


def get_list_of_node_nums():
    node_numbers_with_smote = [
        "504", "506", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "524", "525", "526", "529", "530", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "546", "547", "548", "549", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "581", "582", "584", "585", "586", "587", "588", "589", "590", "591", "592", "593", "594", "595", "596", "598", "599", "600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "627", "628", "629", "630", "632", "633", "634", "635", "636", "637", "638", "639", "640", "641", "642", "643", "644", "645", "646", "647", "648", "649", "650", "651", "652", "655", "656", "657", "658", "659", "660", "661", "662", "663", "664", "665", "666", "668", "669", "670", "671", "672", "673", "674", "675", "676", "677", "678", "681", "683", "685", "686", "690", "691", "692", "693", "694", "695", "696", "697", "698", "699", "700", "701", "702", "703", "704", "705", "706", "707", "708", "709", "710", "711", "712", "713", "714", "715", "716", "717", "718", "719", "720", "721", "722", "723", "724", "725", "726", "727", "728", "730", "731", "732", "733", "735", "736", "737", "738", "739", "740", "741", "742", "743", "744", "745", "746", "747", "748", "749", "750", "751", "756", "757", "758", "759", "760", "761", "762", "763", "764", "765", "766", "767", "770", "771", "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", "787", "788", "789", "790", "791", "792", "793", "795", "796", "797", "798", "799", "800", "801", "802", "803", "804", "805", "806", "808", "809", "810", "811", "812", "813", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829", "830", "831", "832", "834", "835", "836", "837", "838", "839", "841", "842", "843", "844", "845", "846", "847", "848", "849", "850", "851", "852", "853", "854", "855", "856", "857", "858", "859", "860", "861", "862", "863", "864", "865", "866", "867", "868", "869", "870", "871", "872", "873", "874", "875", "877", "878", "879", "880", "881", "882", "883", "885", "886", "887", "888", "889", "890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "915", "916", "917", "918", "919", "920", "921", "922", "923", "924", "925", "926", "927", "928", "929", "930", "931", "932", "933", "934", "935", "937", "938", "939", "940", "941", "942", "943", "944", "945", "946", "947", "948", "949", "950", "951", "952", "953", "954", "955", "956", "957", "958", "960", "961", "962", "963", "964", "965", "968", "969", "970", "971", "973", "974", "975", "976", "977", "978", "979", "980", "981", "982", "983"
    ]

    return node_numbers_with_smote

# list of all 827 nodes for which SMOTE is possible (atleast 1 EZ)
node_numbers_with_smote = get_list_of_node_nums()
# node_numbers_with_smote = ["14", "18", "19"]

print(f"Total Number of nodes in right Hemisphere: {len(node_numbers_with_smote)}")


def load_validation_cohort(root: str, node_num: str = "1"):    
      
    ## Load ValidCohort
    print(f"Loading ValidCohort for Node num: {node_num}")

    path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')
    path_label = os.path.join(root,'New_SOZ_Labels','Orig_Label_Valid')

    RI_file = f"ValidCohort_NonEZvsEZ_RI_node{node_num}_ALL.mat"
    Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
    label_file = f"Valid_Cohort_Label_node{node_num}_ALL.mat"
    
    RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
    Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
    label_mat_name = "Valid_Cohort_Label"

    raw_path_RI = os.path.join(path,RI_file)
    raw_path_Conn = os.path.join(path,Conn_file)
    raw_path_label = os.path.join(path_label,label_file)

    """Load the Relative Intensity (RI) Data Matrix from .mat files.""" 
    X_mat_l = loadmat(raw_path_RI)
    X_mat_RI = X_mat_l[RI_mat_name] # RI matrix: 1x1400

    # check for NaN values and replace NaN values with 0
    if (np.isnan(X_mat_RI).any()):  
        X_mat_RI = np.nan_to_num(X_mat_RI, nan=0) 

    """Load the Connectome Profile (DWIC) Matrix from .mat files.""" 
    X_mat_lconn = loadmat(raw_path_Conn)
    X_mat_DWIC = X_mat_lconn[Conn_mat_name]  # DWIC matrix: 1x499
                
    # check for NaN values and replace NaN values with 0
    if (np.isnan(X_mat_DWIC).any()):
        X_mat_DWIC = np.nan_to_num(X_mat_DWIC, nan=0)

    X_combined_2 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features

    X_combined_2 = X_combined_2[14:,:] # last 10 patients of the validation cohort

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat_aug = Y_mat_l[label_mat_name]
    Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

    Y_mat_aug_2 = Y_mat_aug_2[14:] # last 10 patients of the validation cohort    

    Y_mat_aug_2 = Y_mat_aug_2.astype(int) 

    print('Y_val_node_original: %s' % Counter(Y_mat_aug_2))

    ###################################################################################################
    ## Perform Sub-Group Analysis

    # Load the sub-group analysis file
    sub_group_analysis_file = "Sub_group_analysis.xlsx"
    load_filepath_val = os.path.join(root, sub_group_analysis_file)
    subgroup_file = openpyxl.load_workbook(load_filepath_val)
    subgroup_sheet = subgroup_file['Sheet1']

    ###################################################################
    ## MR subgroups [MR0(MR-: MR=0)/MR1(MR+: MR=1)]

    MR_group = [subgroup_sheet['D60'].value, subgroup_sheet['D61'].value, subgroup_sheet['D62'].value, subgroup_sheet['D63'].value, subgroup_sheet['D64'].value, subgroup_sheet['D65'].value, subgroup_sheet['D66'].value, subgroup_sheet['D67'].value, subgroup_sheet['D68'].value, subgroup_sheet['D69'].value]

    print(f"MR_group: {MR_group}")

    groups_MR_X = {0: [], 1: []}

    for group, row in zip(MR_group, X_combined_2):   
        groups_MR_X[group].append(row)

    X_MR0 = np.array(groups_MR_X[0]) 
    X_MR1 = np.array(groups_MR_X[1])

    groups_MR_Y = {0: [], 1: []}

    for group, row in zip(MR_group, Y_mat_aug_2):   
        groups_MR_Y[group].append(row)

    Y_MR0 = np.array(groups_MR_Y[0]) 
    Y_MR1 = np.array(groups_MR_Y[1])  

    ###################################################################

    ## SF subgroups [SF(SF=1)/SR(SF=0)]

    SF_group = [subgroup_sheet['C60'].value, subgroup_sheet['C61'].value, subgroup_sheet['C62'].value, subgroup_sheet['C63'].value, subgroup_sheet['C64'].value, subgroup_sheet['C65'].value, subgroup_sheet['C66'].value, subgroup_sheet['C67'].value, subgroup_sheet['C68'].value, subgroup_sheet['C69'].value]

    print(f"SF_group: {SF_group}")

    groups_SF_X = {0: [], 1: []}

    for group, row in zip(SF_group, X_combined_2):   
        groups_SF_X[group].append(row)

    X_SR = np.array(groups_SF_X[0]) 
    X_SF = np.array(groups_SF_X[1])

    groups_SF_Y = {0: [], 1: []}

    for group, row in zip(SF_group, Y_mat_aug_2):   
        groups_SF_Y[group].append(row)

    Y_SR = np.array(groups_SF_Y[0]) 
    Y_SF = np.array(groups_SF_Y[1])

    subgroup_file.close() 

    ################################################################################################### 
    # For MR subgroups  
    
    ## Exclude the patients for which label=1, i.e. remove IZ patients for MR0(MR-: MR=0) 
    # find the index of Y_MR0 where Y_MR0 == 1 (IZ patient)
    index_IZ_MR0 = np.where(Y_MR0 == 1)[0]

    # remove the vector of X_MR0 and value of Y_MR0 for IZ
    X_MR0 = np.delete(X_MR0, index_IZ_MR0, axis=0)
    Y_MR0 = np.delete(Y_MR0, index_IZ_MR0, axis=0)

    ## Exclude the patients for which label=1, i.e. remove IZ patients for MR1(MR+: MR=1)
    # find the index of Y_MR1 where Y_MR1 == 1 (IZ patient)
    index_IZ_MR1 = np.where(Y_MR1 == 1)[0]

    # remove the vector of X_MR1 and value of Y_MR1 for IZ
    X_MR1 = np.delete(X_MR1, index_IZ_MR1, axis=0)
    Y_MR1 = np.delete(Y_MR1, index_IZ_MR1, axis=0)

    ###################################################################################################

    # find the index of Y_MR0 where Y_MR0 == 2 or Y_MR0 == 3
    index_SOZ_MR0 = np.where((Y_MR0 == 2) | (Y_MR0 == 3))[0]

    # replace the value of Y_MR0 for index_SOZ with 1
    Y_MR0[index_SOZ_MR0] = 1

    Y_valid_MR0 = Y_MR0.astype(int) # type:ignore

    print('Y_valid_MR0: %s' % Counter(Y_valid_MR0))

    # find the index of Y_MR1 where Y_MR1 == 2 or Y_MR1 == 3
    index_SOZ_MR1 = np.where((Y_MR1 == 2) | (Y_MR1 == 3))[0]

    # replace the value of Y_MR1 for index_SOZ with 1
    Y_MR1[index_SOZ_MR1] = 1

    Y_valid_MR1 = Y_MR1.astype(int) # type:ignore

    print('Y_valid_MR1: %s' % Counter(Y_valid_MR1))
  
    ################################################################################################### 
    # For SF subgroups  
    
    ## Exclude the patients for which label=1, i.e. remove IZ patients for SR(SF=0) 
    # find the index of Y_SR where Y_SR == 1 (IZ patient)
    index_IZ_SR = np.where(Y_SR == 1)[0]

    # remove the vector of X_SR and value of Y_SR for IZ
    X_SR = np.delete(X_SR, index_IZ_SR, axis=0)
    Y_SR = np.delete(Y_SR, index_IZ_SR, axis=0)

    ## Exclude the patients for which label=1, i.e. remove IZ patients for SF(SF=1)
    # find the index of Y_SF where Y_SF == 1 (IZ patient)
    index_IZ_SF = np.where(Y_SF == 1)[0]

    # remove the vector of X_SF and value of Y_SF for IZ
    X_SF = np.delete(X_SF, index_IZ_SF, axis=0)
    Y_SF = np.delete(Y_SF, index_IZ_SF, axis=0)

    ###################################################################################################

    # find the index of Y_SR where Y_SR == 2 or Y_SR == 3
    index_SOZ_SR = np.where((Y_SR == 2) | (Y_SR == 3))[0]

    # replace the value of Y_SR for index_SOZ with 1
    Y_SR[index_SOZ_SR] = 1

    Y_valid_SR = Y_SR.astype(int) # type:ignore

    print('Y_valid_SR: %s' % Counter(Y_valid_SR))

    # find the index of Y_SF where Y_SF == 2 or Y_SF == 3
    index_SOZ_SF = np.where((Y_SF == 2) | (Y_SF == 3))[0]

    # replace the value of Y_SF for index_SOZ with 1
    Y_SF[index_SOZ_SF] = 1

    Y_valid_SF = Y_SF.astype(int) # type:ignore

    print('Y_valid_SF: %s' % Counter(Y_valid_SF))

    ###################################################################################################  

    return X_MR0, Y_valid_MR0, X_MR1, Y_valid_MR1, X_SR, Y_valid_SR, X_SF, Y_valid_SF   


def save_aug_data_as_separate_nodes(save_dir: str, X: np.ndarray, Y: np.ndarray, mode: str = "validation") -> None:
    
    for i in range(len(Y)):
        if mode == "validation":
            savemat(os.path.join(save_dir,'X_valid_orig_patient' + str(i) + '.mat'), {"X_orig_valid_patient" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_valid_orig_patient' + str(i) + '.mat'), {"Y_orig_valid_patient" + str(i):Y[i]})

        else:
            raise KeyError(f"The mode must be either validation.")


def main(root: str, save_path_validation: str): 

    node_numbers = []

    num_nonEZs_val_MR0 = []
    num_EZs_val_MR0 = []

    num_nonEZs_val_MR1 = []
    num_EZs_val_MR1 = []

    num_nonEZs_val_SR = []
    num_EZs_val_SR = []

    num_nonEZs_val_SF = []
    num_EZs_val_SF = []

    for i in node_numbers_with_smote: 
        node_numbers.append(i)

        ## Load and save the original unaugmented validation data        
        X_MR0, Y_MR0, X_MR1, Y_MR1, X_SR, Y_SR, X_SF, Y_SF = load_validation_cohort(root, node_num=i)  # type:ignore

        #####################################################################################
        ## MR0 subgroup (MR-: MR=0)

        # calculate number of non_EZs and EZs for a given original validation node - MR0
        num_nonEZs_val_MR0.append(len(Y_MR0) - np.sum(Y_MR0))
        num_EZs_val_MR0.append(np.sum(Y_MR0))

        # save the original validation data
        save_path_validation = save_path_validation

        save_dir_val_temp = 'Node_' + i
        save_dir_val_temp1 = 'MR0'        
        
        save_dir_val = os.path.join(save_path_validation, save_dir_val_temp, save_dir_val_temp1)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)

        # to save the patients separately
        save_aug_data_as_separate_nodes(save_dir_val, X_MR0, Y_MR0, mode="validation")  # type:ignore

        #####################################################################################
        ## MR1 subgroup (MR+: MR=1)

        # calculate number of non_EZs and EZs for a given original validation node - MR1
        num_nonEZs_val_MR1.append(len(Y_MR1) - np.sum(Y_MR1))
        num_EZs_val_MR1.append(np.sum(Y_MR1))

        # save the original validation data
        save_path_validation = save_path_validation

        save_dir_val_temp = 'Node_' + i
        save_dir_val_temp1 = 'MR1'        
        
        save_dir_val = os.path.join(save_path_validation, save_dir_val_temp, save_dir_val_temp1)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)

        # to save the patients separately
        save_aug_data_as_separate_nodes(save_dir_val, X_MR1, Y_MR1, mode="validation")  # type:ignore

        #####################################################################################
        ## SR subgroup (SF=0)

        # calculate number of non_EZs and EZs for a given original validation node - SR
        num_nonEZs_val_SR.append(len(Y_SR) - np.sum(Y_SR))
        num_EZs_val_SR.append(np.sum(Y_SR))

        # save the original validation data
        save_path_validation = save_path_validation

        save_dir_val_temp = 'Node_' + i
        save_dir_val_temp1 = 'SR'        
        
        save_dir_val = os.path.join(save_path_validation, save_dir_val_temp, save_dir_val_temp1)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)

        # to save the patients separately
        save_aug_data_as_separate_nodes(save_dir_val, X_SR, Y_SR, mode="validation")  # type:ignore

        #####################################################################################
        ## SF subgroup (SF=1)

        # calculate number of non_EZs and EZs for a given original validation node - SF
        num_nonEZs_val_SF.append(len(Y_SF) - np.sum(Y_SF))
        num_EZs_val_SF.append(np.sum(Y_SF))

        # save the original validation data
        save_path_validation = save_path_validation

        save_dir_val_temp = 'Node_' + i
        save_dir_val_temp1 = 'SF'        
        
        save_dir_val = os.path.join(save_path_validation, save_dir_val_temp, save_dir_val_temp1)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)

        # to save the patients separately
        save_aug_data_as_separate_nodes(save_dir_val, X_SF, Y_SF, mode="validation")  # type:ignore

    
    # saving the dataframe
    # path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Hemis/SubGroups/"
    path = "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/SubGroups/"   
    save_path = os.path.join(path, "Information")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # dictionary of lists - MR0
    info_dict_MR0 = {'Node #': node_numbers, 'NonEZ-val-orig': num_nonEZs_val_MR0, 'EZ-val-orig': num_EZs_val_MR0}    

    df = pd.DataFrame(info_dict_MR0)  
    
    filename  = "info_right_hemis_MR0.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    #########################################################################################################

    # dictionary of lists - MR1
    info_dict_MR1 = {'Node #': node_numbers, 'NonEZ-val-orig': num_nonEZs_val_MR1, 'EZ-val-orig': num_EZs_val_MR1}    

    df = pd.DataFrame(info_dict_MR1)  

    # saving the dataframe    
    filename  = "info_right_hemis_MR1.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    #########################################################################################################

    # dictionary of lists - SR
    info_dict_SR = {'Node #': node_numbers, 'NonEZ-val-orig': num_nonEZs_val_SR, 'EZ-val-orig': num_EZs_val_SR}    

    df = pd.DataFrame(info_dict_SR)  

    # saving the dataframe    
    filename  = "info_right_hemis_SR.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    ################################################################################################################

    # dictionary of lists - SF
    info_dict_SF = {'Node #': node_numbers, 'NonEZ-val-orig': num_nonEZs_val_SF, 'EZ-val-orig': num_EZs_val_SF}    

    df = pd.DataFrame(info_dict_SF)  

    # saving the dataframe    
    filename  = "info_right_hemis_SF.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)


if __name__ == "__main__":

    # Root Folder for the dataset
    root='/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/'
    # root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'

    # save_path_validation = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Hemis/SubGroups/'

    save_path_validation = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/SubGroups/'
    
    main(root, save_path_validation)
