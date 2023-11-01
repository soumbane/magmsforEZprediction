import pandas as pd
import os

# Define the paths of your Excel files
# base_path = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Left_Temporal_Lobe/'
base_path = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Temporal_Lobe/'

node_nums = ["385","386","387","388","389","390","391","392","393","394","395","396","397","398","399","400","401","402","403","404","405","406","407","408","409","410","411","412","413","414","415","416","417","418","419","420","421","422","423","424","425","426","427","428","429","430","431","432","433","434","435","436","437","438","439","440","441","442","443","444","445","446","447","448","449","450","451","452","453","454","455","456","458","459","460","461","462","463","464","465","466","467","468","469","470","471","472","473","474","475","476","477","478","479"]

file_paths_val = []

for node_num in node_nums:
    # file_path_val = os.path.join(base_path, "Node_"+node_num+"_Results", "Eval_Results", "results_val_ALL_modalities.xlsx") # For ALL modality combinations
    file_path_val = os.path.join(base_path, "Node_"+node_num+"_Results", "Eval_Results", "results_LeftTemp_val_FULL_Modality_Only.xlsx") # For FULL modality Only
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
combined_df_val.to_excel('LeftTemp_val_FULL_modality_Only.xlsx', index=False)

