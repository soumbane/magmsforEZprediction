import pandas as pd
import os

# Define the paths of your Excel files
base_path = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/'

node_nums = ["888","889","890","891","892","893","894","895",
    "896","897","898","899","900","901","902","903","904","905","906","907","908","909","910","911","912",
    "913","914","915","916","917","918","919","920","921","922","923","924","925","926","927","928","929",
    "930","931","932","933","934","935","936","937","938","939","940","941","942","943","944","945","946",
    "947","948","949","950","951","952","953","954","955","956","957","958","959","960","961",
    "962","963","964","965","966","968","969","970","971","973","974","975","976","977","978","979",
    "980","981","982","983"]

file_paths_val = []
file_paths_train = []

for node_num in node_nums:
    file_path_val = os.path.join(base_path, "Node_"+node_num, "Results", "results_val.xlsx")
    file_paths_val.append(file_path_val)

    file_path_train = os.path.join(base_path, "Node_"+node_num, "Results", "results_train.xlsx")
    file_paths_train.append(file_path_train)


# Initialize an empty DataFrame
combined_df_val = pd.DataFrame()

# Loop through the files and stack the rows
for path in file_paths_val:
    # Load the Excel file
    df = pd.read_excel(path)  

    # Stack the rows vertically
    combined_df_val = pd.concat([combined_df_val, df], axis=0)

# Reset the index to avoid duplicate row indices
combined_df_val = combined_df_val.reset_index(drop=True)

# Save the combined DataFrame to a new Excel file
combined_df_val.to_excel('combined_val.xlsx', index=False)

# Do the same thing for train also
combined_df_train = pd.DataFrame()

for path in file_paths_train:
    # Load the Excel file
    df = pd.read_excel(path)  

    # Stack the rows vertically
    combined_df_train = pd.concat([combined_df_train, df], axis=0)

# Reset the index to avoid duplicate row indices
combined_df_train = combined_df_train.reset_index(drop=True)

# Save the combined DataFrame to a new Excel file
combined_df_train.to_excel('combined_train.xlsx', index=False)
