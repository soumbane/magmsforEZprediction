# Use SMOTE to augment EZ predict dataset to balance both classes and create more samples for training
# Combine all brain nodes node by node (Pat 1 Node 1, Pat 2 Node 1, ...., Pat 44 Node 983): Node Level for Model and Validation Cohort
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

# node_numbers_with_smote = ["134","1","14","3","948"]

print(len(node_numbers_with_smote))

# Combining all nodes node by node after performing SMOTE for each node

# Perform SMOTE augmentation for each node
def augment_data(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 1, num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, random_state: int = 100):

    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
    
    X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore
    
    return X_aug, Y_aug


def train_test_split(X: np.ndarray, Y: np.ndarray, fold: str = "1"):

    if fold == "1": # Last 14 patients for testing/First 30 patients for training
    
        # For the training set (Original patients Only)  
        X_train = X[:30,:]
        Y_train = Y[:30]

        print('Y_train: %s' % Counter(Y_train))

        # For the testing set (Original patients Only)  
        X_test = X[30:44,:]
        Y_test = Y[30:44]

        print('Y_test: %s' % Counter(Y_test))

        print("Fold 1")

    elif fold == "2": # patients 17-30 for testing/First 16 and last 14 patients for training
    
        # For the original training set
        X_train = np.concatenate((X[:16,:], X[30:44,:]), axis=0)
        
        Y_train = np.concatenate((Y[:16], Y[30:44]), axis=0)

        print('Y_train: %s' % Counter(Y_train))        

        # For the original testing set
        X_test = X[16:30,:]
        Y_test = Y[16:30]

        print('Y_test: %s' % Counter(Y_test))

        print("Fold 2")

    elif fold == "3": # last 28 patients for training/First 16 patients for testing
    
        # For the original training set
        X_train = X[16:,:]
        
        Y_train = Y[16:]

        print('Y_train: %s' % Counter(Y_train))

        # For the original testing set    
        X_test = X[:16,:]
        Y_test = Y[:16]

        print('Y_test: %s' % Counter(Y_test))

        print("Fold 3")

    else:
        raise KeyError(f"The fold number must be either 1, 2 or 3.")

    return X_train, Y_train, X_test, Y_test  # type:ignore


def load_model_cohort(root: str, fold_no: str = "1", num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, random_state: int = 100): 

    X_combined_train = np.zeros((1,1899))
    Y_combined_train = []

    X_combined_test = np.zeros((1,1899))
    Y_combined_test = []

    for i in node_numbers_with_smote:      

        ## Load ModelCohort
        print(f"Loading ModelCohort for Node num: {i}")

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
        Y_mat_aug_1 = Y_mat_aug_1.astype(int) # type:ignore

        print(f"Combined node {i} of 44 Model Cohort patients.")

        # split the model cohort into training and testing sets
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = train_test_split(X_combined_1, Y_mat_aug_1, fold=fold_no) # type:ignore

        if np.sum(Y_train_orig) == 0:
            # cannot perform SMOTE NO samples for EZ (class 1) is present - just append original node data to whole brain data
            X_train, Y_train = X_train_orig, Y_train_orig

        elif np.sum(Y_train_orig) == 1:
            # cannot perform SMOTE as only 1 sample for EZ (class 1) is present - No nearest neighbors
            # just append original node data to whole brain data
            X_train, Y_train = X_train_orig, Y_train_orig

        elif np.sum(Y_train_orig) == 2:
            # augment training data using SMOTE with KNN=1 (balance training dataset with 1 nearest neighbor)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=1, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore
        
        elif np.sum(Y_train_orig) == 3:
            # augment training data using SMOTE with KNN=2 (balance training dataset with 2 nearest neighbors)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=2, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore
        
        elif np.sum(Y_train_orig) == 4:
            # augment training data using SMOTE with KNN=3 (balance training dataset with 3 nearest neighbors)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=3, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore

        elif np.sum(Y_train_orig) == 5:
            # augment training data using SMOTE with KNN=4 (balance training dataset with 4 nearest neighbors)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=4, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore

        elif np.sum(Y_train_orig) == 6:
            # augment training data using SMOTE with KNN=5 (balance training dataset with 5 nearest neighbors)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=5, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore

        else:
            # augment training data using SMOTE with KNN=6 (balance training dataset with 6 nearest neighbors)
            X_train, Y_train = augment_data(X_train_orig, Y_train_orig, k_neighbors=6, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state) # type:ignore

        # Combine the original+augmented training data for node i
        X_combined_train = np.concatenate((X_combined_train, X_train), axis=0)
        if i == node_numbers_with_smote[0]:
            X_combined_train = X_combined_train[1:,:]

        Y_combined_train = np.concatenate((Y_combined_train, Y_train), axis=0)
        Y_combined_train = Y_combined_train.astype(int) # type:ignore

        # Combine the original testing data for node i
        X_combined_test = np.concatenate((X_combined_test, X_test_orig), axis=0)
        if i == node_numbers_with_smote[0]:
            X_combined_test = X_combined_test[1:,:]

        Y_combined_test = np.concatenate((Y_combined_test, Y_test_orig), axis=0)
        Y_combined_test = Y_combined_test.astype(int) # type:ignore

        print("Finished Model Cohort")

    if fold_no == "1":
        print(f"Finished combining all {len(node_numbers_with_smote)} nodes of first 30 training patients and last 14 test patients (Fold 1) from Model Cohort.")
    elif fold_no == "2":
        print(f"Finished combining all {len(node_numbers_with_smote)} nodes of first 16 and last 14 training patients and 17-30 test patients (Fold 2) from Model Cohort.")
    elif fold_no == "3":
        print(f"Finished combining all {len(node_numbers_with_smote)} nodes of last 28 training patients and first 16 test patients (Fold 3) from Model Cohort.")
    else:
        raise KeyError(f"The fold number must be either 1, 2 or 3.")

    return X_combined_train, Y_combined_train, X_combined_test, Y_combined_test


def load_validation_cohort(root: str):    
          
    X_combined_whole_brain = np.zeros((1,1899))
    Y_combined_whole_brain = []

    for i in node_numbers_with_smote:      
        ## Load ValidCohort
        print(f"Loading ValidCohort for Node num: {i}")

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
        if i == node_numbers_with_smote[0]:
            X_combined_whole_brain = X_combined_whole_brain[1:,:]           
                        
        Y_combined_whole_brain = np.concatenate((Y_combined_whole_brain, Y_mat_aug_2), axis=0)
        Y_combined_whole_brain = Y_combined_whole_brain.astype(int) 
        print(f"Combined node {i} of 24 Validation Cohort patients.")

    print(f"Finished combining all {len(node_numbers_with_smote)} nodes of 24 patients of the independent Validation Cohort.")

    return X_combined_whole_brain, Y_combined_whole_brain


def save_aug_data_as_separate_nodes(save_dir: str, X: np.ndarray, Y: np.ndarray, mode: str = "train") -> None:
    
    for i in range(len(Y)):
        if mode == "train":
            savemat(os.path.join(save_dir,'X_train_aug_WB_node' + str(i) + '.mat'), {"X_aug_train_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_train_aug_WB_node' + str(i) + '.mat'), {"Y_aug_train_node" + str(i):Y[i]})

        elif mode == "test":
            savemat(os.path.join(save_dir,'X_test_orig_WB_node' + str(i) + '.mat'), {"X_orig_test_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_test_orig_WB_node' + str(i) + '.mat'), {"Y_orig_test_node" + str(i):Y[i]})

        elif mode == "validation":
            savemat(os.path.join(save_dir,'X_valid_orig_WB_node' + str(i) + '.mat'), {"X_orig_valid_node" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_valid_orig_WB_node' + str(i) + '.mat'), {"Y_orig_valid_node" + str(i):Y[i]})

        else:
            raise KeyError(f"The mode must be either train, test or validation.")


def main(root: str, save_path_training: str, save_path_testing: str, save_path_validation: str, fold_no: str = "1", num_samples_nonEZ: int = 50, num_samples_EZ: int = 50):    
    
    ## Load and save the Model cohort data (44 patients) divided into training (30 patients) and testing (14 patients)
    # Combining node by node (Pat 1 Node 1, Pat 2 Node 1, ...., Pat 44 Node 983): Node Level for Model Cohort
    X_combined_train, Y_combined_train, X_combined_test, Y_combined_test = load_model_cohort(root, fold_no=fold_no, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=100)  # type:ignore    

    print('Y_combined_train: %s' % Counter(Y_combined_train))

    print('Y_combined_test: %s' % Counter(Y_combined_test))
       
    # save the augmented training data
    save_path_training = save_path_training

    save_dir_train_temp = 'Train_NonEZvsEZ_WB_smoteaug_fold' + fold_no
    save_dir_train = os.path.join(save_path_training, save_dir_train_temp)

    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    
    save_aug_data_as_separate_nodes(save_dir_train, X_combined_train, Y_combined_train, mode="train")  # type:ignore  

    # save the original testing data
    save_path_testing = save_path_testing

    save_dir_test_temp = 'Test_NonEZvsEZ_WB_orig_fold' + fold_no
    save_dir_test = os.path.join(save_path_testing, save_dir_test_temp)
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    
    save_aug_data_as_separate_nodes(save_dir_test, X_combined_test, Y_combined_test, mode="test")  # type:ignore  

    ################################################################################################################
    ## Load and save the original independent (Hold-out) validation cohort data (24 patients)
    
    # Combining node by node (Pat 1 Node 1, Pat 2 Node 1, ...., Pat 24 Node 983): Node Level for Validation Cohort
    X_combined_validation, Y_combined_validation = load_validation_cohort(root)  # type:ignore

    print('Y_combined_validation: %s' % Counter(Y_combined_validation))

    save_path_validation = save_path_validation
    
    save_dir_val = os.path.join(save_path_validation, 'ValidationCohort_NonEZvsEZ_WB_orig')
    if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)

    save_aug_data_as_separate_nodes(save_dir_val, X_combined_validation, Y_combined_validation, mode="validation")  # type:ignore

    ################################################################################################################


if __name__ == "__main__":

    # Root Folder for the dataset
    # root='/home/user1/Desktop/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/'
    root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'

    save_path_training = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/NodebyNode_Data/SMOTE_Augmented_Data/'
    save_path_testing = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/NodebyNode_Data/Original_Patient_Data/'
    save_path_validation = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/NodebyNode_Data/Original_Patient_Data/'

    # num_samples_nonEZ: Number of samples of non-EZ (class 0) to generate per node with SMOTE
    # num_samples_EZ: Number of samples of EZ (class 1) to generate per node with SMOTE

    main(root, save_path_training, save_path_testing, save_path_validation, fold_no="1", num_samples_nonEZ=30, num_samples_EZ=40) # for trying

    # main(root, save_path_training, save_path_testing, save_path_validation, fold_no="1", num_samples_nonEZ=80, num_samples_EZ=100)
    # main(root, save_path_training, save_path_testing, save_path_validation, fold_no="2", num_samples_nonEZ=80, num_samples_EZ=100)
    # main(root, save_path_training, save_path_testing, save_path_validation, fold_no="3", num_samples_nonEZ=80, num_samples_EZ=100)


