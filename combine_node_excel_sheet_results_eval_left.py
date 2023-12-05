import pandas as pd
import os

# Define the paths of your Excel files
# base_path = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Left_Temporal_Lobe/Part_2/'
# base_path = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Temporal_Lobe/Part_2/'
base_path = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/'

node_nums = ["6", "11", "12", "14", "18", "19", "20", "33", "34", "35", "36", "39", "41", "42", "43", "44", "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "66", "68", "79", "80", "81", "84", "85", "86", "87", "88", "90", "91", "93", "94", "95", "96", "97", "98", "101", "102", "103", "104", "108", "109", "111", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "140", "144", "145", "147", "148", "150", "151", "155", "156", "158", "159", "160", "163", "164", "165", "166", "169", "175", "176", "177", "192", "193", "194", "195", "197", "198", "199", "200", "202", "204", "205", "211", "213", "214", "216", "217", "220", "221", "222", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "238", "239", "240", "241", "245", "246", "247", "248", "251", "252", "253", "256", "257", "260", "261", "275", "290", "291", "292", "294", "295", "296", "297", "298", "299", "301", "302", "303", "304", "305", "306", "316", "320", "321", "322", "326", "331", "332", "334", "335", "336", "337", "338", "339", "340", "343", "346", "349", "352", "353", "354", "355", "356", "357", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "374", "375", "376", "377", "378", "381", "382", "383", "384", "386", "387", "388", "389", "390", "391", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "426", "427", "428", "429", "430", "431", "432", "433", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "450", "451", "452", "453", "454", "455", "456", "458", "459", "460", "461", "462", "463", "464", "465", "466", "467", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479"] # part 2

file_paths_val = []

for node_num in node_nums:
    # file_path_val = os.path.join(base_path, "Node_"+node_num+"_Results", "Eval_Results", "results_val_ALL_modalities_Part_2.xlsx") # For ALL modality combinations
    file_path_val = os.path.join(base_path, "Node_"+node_num+"_Results", "Eval_Results", "results_LeftHemis_val_FULL_Modality_Only_Part_2.xlsx") # For FULL modality Only
    file_paths_val.append(file_path_val)

# Initialize an empty DataFrame
combined_df_val = pd.DataFrame()

# Loop through the files and stack the rows
for path in file_paths_val:
    # Load the Excel file
    df = pd.read_excel(path)  

    # Stack the rows
    # combined_df_val = pd.concat([combined_df_val, df], axis=1) # For ALL modality combinations
    combined_df_val = pd.concat([combined_df_val, df], axis=0) # For FULL modality Only

# Reset the index to avoid duplicate row indices
combined_df_val = combined_df_val.reset_index(drop=True)

# Save the combined DataFrame to a new Excel file
combined_df_val.to_excel('LeftHemis_val_FULL_modality_Only_Part_2.xlsx', index=False)
# combined_df_val.to_excel('LeftHemis_val_ALL_modality_Comb_Part_2.xlsx', index=False)

