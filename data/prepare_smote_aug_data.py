# Use SMOTE to augment EZ predict dataset to balance both classes and create more samples for the whole brain (comibining all the nodes)
import os
import numpy as np
from collections import Counter
from scipy.io import loadmat, savemat
from imblearn.over_sampling import SMOTE


def get_list_of_node_nums():
    node_numbers_with_smote = [
        "1","2","3","5","6","11","12","13","14","17","18","19","20",
        "29","30","33","34","35","36","37","38","39","41","42","43","44","45",
        "46","47","48","49","50","51","52","53","54","55","56","57","58","59","60",
        "61","62","63","64","65","66","67","68","69","70","71","72","77","78","79","80",
        "81","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98",
        "99","100","101","102","103","104","105","107","108","109","110","111","112",
        "113","120","121","122","123","124","125","126","127","128","129","130","131","132","133",
        "134","135","136","137","140","141","144","145","146","147","148","149","150",
        "151","152","154","155","156","157","158","159","160","162","163","164","165","166","167",
        "168","169","170","175","176","177","185","187","191","192","193","194","195",
        "196","197","198","199","200","201","202","204","205","208","209","210","211",
        "212","213","214","215","216","217","220","221","222","224","225","226","227","228","229",
        "230","231","232","233","234","235","238","239","240","241","243","244","245","246","247",
        "248","249","250","251","252","253","254","255","256","257","259","260","261","262","263",
        "264","275","287","288","289","290","291","292","294","295","296","297","298","299","300",
        "301","302","303","304","305","306","313","316","320","321","322","325","326","327",
        "331","332","334","335","336","337","338","339","340","341","343","346","349","352",
        "353","354","355","356","357","359","360","361","362","363","364","365","366","367","368","369",
        "370","371","372","373","374","375","376","377","378","381","382","383","384","385",
        "386","387","388","389","390","391","392","393","394","395","396","397","398","399","400",
        "401","402","403","404","405","406","407","408","409","410","411","412","413","414","415","416","417",
        "418","419","420","421","422","423","424","425","426","427","428","429","430","431","432",
        "433","434","435","436","437","438","439","440","441","442","443","444","445","446","447","448",
        "449","450","451","452","453","454","455","456","458","459","460","461","462","463","464","465",
        "466","467","468","469","470","471","472","473","474","475","476","477","478","479","500","501",
        "502","503","504","505","506","507","508","509","510","511","512","513","514","515","516","517",
        "518","519","520","521","522","524","525","526","529","530","531","532","533","534","535",
        "536","537","538","539","540","541","542","543","544","545","546","547","548","549","550","551","552",
        "553","554","555","556","557","558","559","560","561","562","563","564","565","566","567","568",
        "569","570","571","572","573","574","575","576","577","578","579","581","582","583","584",
        "585","586","587","588","589","590","591","592","593","594","595","596","597","598","599",
        "600","601","602","603","604","605","606","607","608","609","610","611","612","613","614","615","616","617","618","619","620","621","622","623","624","625","626","627","628","629","630","631","632","633","634","635","636","637","638","639","640","641","642","643","644","645","646","647","648","649","650","651","652","653","655","656","657","658","659","660","661","662","663","664","665","666","667","668",
        "669","670","671","672","673","674","675","676","677","678","679","680","681","682","683",
        "685","686","687","688","690","691","692","693","694","695","696","697","698","699","700","701",
        "702","703","704","705","706","707","708","709","710","711","712","713","714","715","716","717",
        "718","719","720","721","722","723","724","725","726","727","728","730","731","732","733",
        "735","736","737","738","739","740","741","742","743","744","745","746","747","748","749","750",
        "751","756","757","758","759","760","761","762","763","764","765","766","767","769","770",
        "771","776","777","778","779","780","781","782","783","784","785","786","787","788",
        "789","790","791","792","793","794","795","796","797","798","799","800","801","802","803","804",
        "805","806","807","808","809","810","811","812","813","816","817","818","819","820",
        "821","822","823","824","825","826","827","828","829","830","831","832","834","835","836",
        "837","838","839","841","842","843","844","845","846","847","848","849","850","851",
        "852","853","854","855","856","857","858","859","860","861","862","863","864","865","866",
        "867","868","869","870","871","872","873","874","875","876","877","878","879","880",
        "881","882","883","885","886","887","888","889","890","891","892","893","894","895",
        "896","897","898","899","900","901","902","903","904","905","906","907","908","909","910","911","912",
        "913","914","915","916","917","918","919","920","921","922","923","924","925","926","927","928","929",
        "930","931","932","933","934","935","936","937","938","939","940","941","942","943","944","945","946",
        "947","948","949","950","951","952","953","954","955","956","957","958","959","960","961",
        "962","963","964","965","966","968","969","970","971","973","974","975","976","977","978","979",
        "980","981","982","983"
    ]

    return node_numbers_with_smote

# list of all 827 nodes for which SMOTE is possible (atleast 1 EZ)
node_numbers_with_smote = get_list_of_node_nums()

# node_numbers_with_smote = ["1","3","948"]

print(len(node_numbers_with_smote))

# New code for both hemisphere (combining all nodes)

def load_data(root: str, mode: str = "model"):
    
    if mode == "all":        
        X_combined_whole_brain = np.zeros((1,1899))
        Y_combined_whole_brain = []

        for i in node_numbers_with_smote:      
            ## Load ModelCohort
            # print(f"Loading ModelCohort for Node num: {i}")

            path = os.path.join(root,'Valid_NonEZvsEZ_ALL')

            RI_file = f"Valid_NonEZvsEZ_RI_node{i}_ALL.mat"
            Conn_file = f"Valid_NonEZvsEZ_Conn_node{i}_ALL.mat"
            label_file = f"Valid_NonEZvsEZ_label_node{i}_ALL.mat"
            RI_mat_name = "ModelCohort_NonEZvsEZ_RI"
            Conn_mat_name = "ModelCohort_NonEZvsEZ_Conn"
            label_mat_name = "ModelCohort_NonEZvsEZ_label"

            raw_path_RI = os.path.join(path,RI_file)
            raw_path_Conn = os.path.join(path,Conn_file)
            raw_path_label = os.path.join(path,label_file)

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

            X_combined_1 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features

            """Load the Label Matrix from .mat files.""" 
            Y_mat_l = loadmat(raw_path_label)
            Y_mat_aug = Y_mat_l[label_mat_name]
            Y_mat_aug_1 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

            X_combined_whole_brain = np.concatenate((X_combined_whole_brain, X_combined_1), axis=0)
            if i == node_numbers_with_smote[0]:
                X_combined_whole_brain = X_combined_whole_brain[1:,:]

            Y_combined_whole_brain = np.concatenate((Y_combined_whole_brain, Y_mat_aug_1), axis=0)

            ###################################################################################################
            ## Load ValidCohort
            # print(f"Loading ValidCohort for Node num: {i}")

            path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')
        
            RI_file = f"ValidCohort_NonEZvsEZ_RI_node{i}_ALL.mat"
            Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{i}_ALL.mat"
            label_file = f"ValidCohort_NonEZvsEZ_label_node{i}_ALL.mat"
            RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
            Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
            label_mat_name = "ValidCohort_NonEZvsEZ_label"

            raw_path_RI = os.path.join(path,RI_file)
            raw_path_Conn = os.path.join(path,Conn_file)
            raw_path_label = os.path.join(path,label_file)

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

            """Load the Label Matrix from .mat files.""" 
            Y_mat_l = loadmat(raw_path_label)
            Y_mat_aug = Y_mat_l[label_mat_name]
            Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

            X_combined_whole_brain = np.concatenate((X_combined_whole_brain, X_combined_2), axis=0)            
                            
            Y_combined_whole_brain = np.concatenate((Y_combined_whole_brain, Y_mat_aug_2), axis=0)
            Y_combined_whole_brain = Y_combined_whole_brain.astype(int) 
            print(f"Combined node {i} of 68 patients.")

        print("Finished combining all 827 nodes of 68 patients.")

        return X_combined_whole_brain, Y_combined_whole_brain 

    elif mode == "patient_level":        
        X_combined_whole_brain = np.zeros((1,1899))
        Y_combined_whole_brain = []

        for j in range(68):
            for i in node_numbers_with_smote:   
                if j < 44:   
                    ## Load ModelCohort
                    # print(f"Loading ModelCohort for Node num: {i}")

                    path = os.path.join(root,'Valid_NonEZvsEZ_ALL')

                    RI_file = f"Valid_NonEZvsEZ_RI_node{i}_ALL.mat"
                    Conn_file = f"Valid_NonEZvsEZ_Conn_node{i}_ALL.mat"
                    label_file = f"Valid_NonEZvsEZ_label_node{i}_ALL.mat"
                    RI_mat_name = "ModelCohort_NonEZvsEZ_RI"
                    Conn_mat_name = "ModelCohort_NonEZvsEZ_Conn"
                    label_mat_name = "ModelCohort_NonEZvsEZ_label"

                    raw_path_RI = os.path.join(path,RI_file)
                    raw_path_Conn = os.path.join(path,Conn_file)
                    raw_path_label = os.path.join(path,label_file)

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

                    X_combined_1 = np.concatenate((X_mat_RI, X_mat_DWIC), axis=1) # using both RI and Conn features

                    """Load the Label Matrix from .mat files.""" 
                    Y_mat_l = loadmat(raw_path_label)
                    Y_mat_aug = Y_mat_l[label_mat_name]
                    # Y_mat_aug_1 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)
                    Y_mat_aug_1 = Y_mat_aug[j,:]

                    X_mat_temp = X_combined_1[j,:].reshape(1, X_combined_1.shape[1])

                    X_combined_whole_brain = np.concatenate((X_combined_whole_brain, X_mat_temp), axis=0)
                    if j == 0 and i == node_numbers_with_smote[0]:
                        X_combined_whole_brain = X_combined_whole_brain[1:,:]

                    Y_combined_whole_brain = np.concatenate((Y_combined_whole_brain, Y_mat_aug_1), axis=0)
                    print(f"ModelCohort: Patient {j+1}, Node {i}")

                ###################################################################################################
                else:
                    ## Load ValidCohort
                    # print(f"Loading ValidCohort for Node num: {i}")

                    path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')
                
                    RI_file = f"ValidCohort_NonEZvsEZ_RI_node{i}_ALL.mat"
                    Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{i}_ALL.mat"
                    label_file = f"ValidCohort_NonEZvsEZ_label_node{i}_ALL.mat"
                    RI_mat_name = "ValidCohort_NonEZvsEZ_RI"
                    Conn_mat_name = "ValidCohort_NonEZvsEZ_Conn"
                    label_mat_name = "ValidCohort_NonEZvsEZ_label"

                    raw_path_RI = os.path.join(path,RI_file)
                    raw_path_Conn = os.path.join(path,Conn_file)
                    raw_path_label = os.path.join(path,label_file)

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

                    """Load the Label Matrix from .mat files.""" 
                    Y_mat_l = loadmat(raw_path_label)
                    Y_mat_aug = Y_mat_l[label_mat_name]
                    # Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

                    Y_mat_aug_2 = Y_mat_aug[j-44,:]

                    X_mat_temp = X_combined_2[j-44,:].reshape(1, X_combined_2.shape[1])

                    X_combined_whole_brain = np.concatenate((X_combined_whole_brain, X_mat_temp), axis=0)
                    
                    Y_combined_whole_brain = np.concatenate((Y_combined_whole_brain, Y_mat_aug_2), axis=0)
                    print(f"ValidationCohort: Patient {(j)+1}, Node {i}")                    

        Y_combined_whole_brain = Y_combined_whole_brain.astype(int) # type:ignore
        print("Finished combining all 827 nodes of 68 patients.")

        return X_combined_whole_brain, Y_combined_whole_brain 

    else:
        raise NotImplementedError(f"Mode must be either model or valid to load dataset.")


def z_score_norm(X: np.ndarray):

    X_mean = np.mean(X, axis=0, dtype=np.float64)
    X_std = np.std(X, axis=0, dtype=np.float64)

    X_norm = (X - X_mean)/X_std

    # check for NaN values and replace NaN values with 0
    if (np.isnan(X_norm).any()):
        X_norm = np.nan_to_num(X_norm, nan=0) 
    
    return X_norm


def augment_data(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5, random_state: int = 100):

    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state) # type:ignore
    
    X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore
    
    return X_aug, Y_aug


'''def augment_data(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5, num_samples: int = 100, random_state: int = 100):

    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state, sampling_strategy={0:num_samples, 1:num_samples}) # type:ignore
    
    X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore
    
    return X_aug, Y_aug'''


def train_val_split(X: np.ndarray, Y: np.ndarray, fold: str = "1", num_nodes: int = 827):

    if fold == "1": # Last 14 patients for validation/First 54 patients for training
    
        # For the training set and its augmentations
        total_ones_train = []
        for i in range(54):
            total_ones_train.append(sum(Y[num_nodes*i:num_nodes*(i+1)])) 

        original_ones_train = sum(total_ones_train)
        original_zeros_train = len(total_ones_train)*num_nodes - sum(total_ones_train)

        augmented_ones_train = original_zeros_train - original_ones_train

        X_train = np.concatenate((X[:54*num_nodes,:], X[68*num_nodes:(68*num_nodes+augmented_ones_train),:]), axis=0)
        Y_train = np.concatenate((Y[:54*num_nodes], Y[68*num_nodes:(68*num_nodes+augmented_ones_train)]), axis=0)

        print('Y_train: %s' % Counter(Y_train))

        # For the validation set and its augmentations
        total_ones_val = []
        for i in range(54,68):
            total_ones_val.append(sum(Y[num_nodes*i:num_nodes*(i+1)]))  

        original_ones_val = sum(total_ones_val)
        original_zeros_val = len(total_ones_val)*num_nodes - sum(total_ones_val)

        augmented_ones_val = original_zeros_val - original_ones_val

        X_val = np.concatenate((X[54*num_nodes:68*num_nodes,:], X[(68*num_nodes+augmented_ones_train):,:]), axis=0)
        Y_val = np.concatenate((Y[54*num_nodes:68*num_nodes], Y[(68*num_nodes+augmented_ones_train):]), axis=0)

        print('Y_val: %s' % Counter(Y_val))


    return X_train, Y_train, X_val, Y_val  # type:ignore


def save_aug_data_as_separate_nodes(save_dir: str, X: np.ndarray, Y: np.ndarray, mode: str = "train") -> None:
    
    for i in range(len(Y)):
        if mode == "train":
            savemat(os.path.join(save_dir,'X_train_aug_whole_brain_node' + str(i) + '.mat'), {"X_aug_train_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_train_aug_whole_brain_node' + str(i) + '.mat'), {"Y_aug_train_node" + str(i):Y[i]})

        elif mode == "valid":
            savemat(os.path.join(save_dir,'X_val_aug_whole_brain_node' + str(i) + '.mat'), {"X_aug_valid_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_val_aug_whole_brain_node' + str(i) + '.mat'), {"Y_aug_valid_node" + str(i):Y[i]})

        else:
            raise KeyError(f"The mode must be either train, valid.")


def main(root: str, k_neighbors: int = 5, num_nodes: int = 3, fold_no: str = "1"):
    
    # Combining all brain nodes patient by patient (Pat 1 Node 1, Pat 1 Node 2, ...., Pat 68 Node 948)
    X_combined_all_patients, Y_combined_all_patients = load_data(root, mode="patient_level")  # type:ignore 

    # Perform z-score normalization across patients
    X_combined_all_patients_norm = z_score_norm(X_combined_all_patients)  # type:ignore
    print(f"X_all_patients max: {np.max(X_combined_all_patients_norm)}")
    print(f"X_all_patients min: {np.min(X_combined_all_patients_norm)}")
    Y_combined_all_patients = Y_combined_all_patients

    # augment data using SMOTE (balance dataset) for all 827 nodes of 68 patients
    X_aug_all_patients, Y_aug_all_patients = augment_data(X_combined_all_patients_norm, Y_combined_all_patients, k_neighbors = k_neighbors, random_state=100) # type:ignore

    print('UnAugmented Y_all_patients shape %s' % Counter(Y_combined_all_patients))
    print('Augmented Y_all_patients shape %s' % Counter(Y_aug_all_patients))

    # split the data into training and validation (80%-20% split) 
    X_train, Y_train, X_val, Y_val = train_val_split(X_aug_all_patients, Y_aug_all_patients, fold=fold_no, num_nodes=num_nodes) # type:ignore 
        
    # Randomly shuffle X_train, Y_train with the same seed
    np.random.seed(0)
    np.random.shuffle(X_train)

    np.random.seed(0)
    np.random.shuffle(Y_train)

    # Randomly shuffle X_val, Y_val with the same seed
    np.random.seed(0)
    np.random.shuffle(X_val)

    np.random.seed(0)
    np.random.shuffle(Y_val)
    
    # save the augmented data
    save_dir_train = 'Train_NonEZvsEZ_whole_brain_aug_separate_fold' + fold_no
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    
    save_aug_data_as_separate_nodes(save_dir_train, X_train, Y_train, mode="train")  # type:ignore  

    save_dir_val = 'Val_NonEZvsEZ_whole_brain_aug_separate_fold' + fold_no
    if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)

    save_aug_data_as_separate_nodes(save_dir_val, X_val, Y_val, mode="valid")  # type:ignore


if __name__ == "__main__":

    # Root Folder
    root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'

    main(root, k_neighbors=6, num_nodes=827, fold_no="1")


