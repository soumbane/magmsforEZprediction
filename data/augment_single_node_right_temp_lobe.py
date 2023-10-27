# Use SMOTE to augment EZ dataset to balance both classes and create more samples for training
# Combine all brain nodes node by node (Pat 1 Node 1, Pat 2 Node 1, ...., Pat 44 Node 983): Node Level for Model and Validation Cohort
import os
import numpy as np
import pandas as pd
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

print(f"Total number of nodes is: {len(node_numbers_with_smote)}")

# temporal lobe of right hemisphere
node_number_right_temporal_lobe = [
    "888","889","890","891","892","893","894","895",
    "896","897","898","899","900","901","902","903","904","905","906","907","908","909","910","911","912",
    "913","914","915","916","917","918","919","920","921","922","923","924","925","926","927","928","929",
    "930","931","932","933","934","935","936","937","938","939","940","941","942","943","944","945","946",
    "947","948","949","950","951","952","953","954","955","956","957","958","959","960","961",
    "962","963","964","965","966","968","969","970","971","973","974","975","976","977","978","979",
    "980","981","982","983"
    ]

node_numbers_with_smote = node_number_right_temporal_lobe

print(f"Number of nodes in right temporal lobe: {len(node_numbers_with_smote)}")

fusiform_ROI = ["888","889","890","891","892","893","894","895","896","897","898","899","900","901","902","903","904"]

parahipp_ROI = ["905","906","907","908","909","910"]

entor_ROI = ["911","912"]

temppole_ROI = ["913","914","915"]

inferiortemp_ROI = ["916","917","918","919","920","921","922","923","924","925","926","927","928","929","930","931"]

middletemp_ROI = ["932","933","934","935","936","937","938","939","940","941","942","943","944","945","946","947","948","949","950"]

bank_ROI = ["951","952","953","954","955","956"]

superiortemp_ROI = ["957","958","959","960","961","962","963","964","965","966","968","969","970","971","973","974","975","976","977","978","979","980","981"]

transversetemp_ROI = ["982","983"]

def load_and_average_ROI_data(root: str, ROI_node_num_range: list[str]):

    X_train_node = np.zeros((1,1899)) 
    Y_train_node = []
    
    for i in ROI_node_num_range:

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

        X_combined_2 = X_combined_2[:14,:] # first 14 patients of the validation cohort

        """Load the Label Matrix from .mat files.""" 
        Y_mat_l = loadmat(raw_path_label)
        Y_mat_aug = Y_mat_l[label_mat_name]
        Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

        Y_mat_aug_2 = Y_mat_aug_2[:14] # first 14 patients of the validation cohort

                    
        # combine training node-level data
        X_train_node = np.concatenate((X_train_node, X_combined_1, X_combined_2), axis=0)
        
        if i == ROI_node_num_range[0]:
            X_train_node = X_train_node[1:,:] 

        Y_train_node = np.concatenate((Y_train_node, Y_mat_aug_1, Y_mat_aug_2), axis=0)

    Y_train_node = Y_train_node.astype(int) # type:ignore
        
    print('Y_train_node: %s' % Counter(Y_train_node))

    # find the index of Y_train_node where Y_train_node == 1 and Y_train_node == 0
    index_EZ = np.where(Y_train_node == 1)[0]

    # get the vector of X_train_node for EZ and non-EZ
    X_train_node_EZ = X_train_node[index_EZ,:]

    # find the average of all EZ and non-EZ vectors
    X_train_node_EZ_avg = np.mean(X_train_node_EZ, axis=0)

    return X_train_node_EZ_avg 


# Augment node by node by performing SMOTE for each node

def augment_1EZ_data(root: str, X: np.ndarray, Y: np.ndarray, num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, random_state: int = 100, generate_syn_nonEZ: bool = True, node_num: str = "1"):

    if node_num in fusiform_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in fusiform ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, fusiform_ROI)
    elif node_num in parahipp_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in parahippocampal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, parahipp_ROI)
    elif node_num in entor_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in entorhinal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, entor_ROI)
    elif node_num in temppole_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in temppole ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, temppole_ROI)
    elif node_num in inferiortemp_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in inferiortemporal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, inferiortemp_ROI)
    elif node_num in middletemp_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in middletemporal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, middletemp_ROI)
    elif node_num in bank_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in bank ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, bank_ROI)
    elif node_num in superiortemp_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in superiortemporal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, superiortemp_ROI)
    elif node_num in transversetemp_ROI:
        print(f"Node-number {node_num} with 1 EZ class is in transversetemporal ROI.")
        X_train_node_EZ_avg = load_and_average_ROI_data(root, transversetemp_ROI)
    else:
        raise ValueError("Node number not found in any ROI.")

    # stack the X vectors with the average ROI vectors
    X_train_node = np.vstack((X, X_train_node_EZ_avg)) 
    Y_train_node = np.append(Y,[1]) 
    
    # augment training data using SMOTE with KNN=1 using the ROI average data
    if generate_syn_nonEZ:
        sm = SMOTE(k_neighbors=1, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
    else: # do not generate synthetic non-EZ samples
        sm = SMOTE(k_neighbors=1, random_state=random_state) # type:ignore

    X_aug, Y_aug = sm.fit_resample(X_train_node, Y_train_node) # type:ignore

    return X_aug, Y_aug


# Perform SMOTE augmentation for each node
def augment_data(X: np.ndarray, Y: np.ndarray, num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, random_state: int = 100, generate_syn_nonEZ: bool = True, node_num: str = "1"):

    if np.sum(Y) == 1:
        print(f"Node-number {node_num} has only 1 EZ sample, hence it should be augmented with ROI-average data.")
        raise ValueError("Cannot augment data for a single EZ sample.")

    elif np.sum(Y) == 2:
        # augment training data using SMOTE with KNN=1
        if generate_syn_nonEZ:
            sm = SMOTE(k_neighbors=1, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
        else: # do not generate synthetic non-EZ samples
            sm = SMOTE(k_neighbors=1, random_state=random_state) # type:ignore
        X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore
    
    elif np.sum(Y) == 3:
        # augment training data using SMOTE with KNN=2
        if generate_syn_nonEZ:
            sm = SMOTE(k_neighbors=2, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
        else: # do not generate synthetic non-EZ samples
            sm = SMOTE(k_neighbors=2, random_state=random_state) # type:ignore
        X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore

    elif np.sum(Y) == 4:
        # augment training data using SMOTE with KNN=3
        if generate_syn_nonEZ:
            sm = SMOTE(k_neighbors=3, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
        else: # do not generate synthetic non-EZ samples
            sm = SMOTE(k_neighbors=3, random_state=random_state) # type:ignore
        X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore

    elif np.sum(Y) == 5:
        # augment training data using SMOTE with KNN=4
        if generate_syn_nonEZ:
            sm = SMOTE(k_neighbors=4, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
        else: # do not generate synthetic non-EZ samples
            sm = SMOTE(k_neighbors=4, random_state=random_state) # type:ignore
        X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore

    else:
        # augment training data using SMOTE with KNN=5
        if generate_syn_nonEZ:
            sm = SMOTE(k_neighbors=5, random_state=random_state, sampling_strategy={0:num_samples_nonEZ, 1:num_samples_EZ}) # type:ignore
        else: # do not generate synthetic non-EZ samples
            sm = SMOTE(k_neighbors=5, random_state=random_state) # type:ignore
        X_aug, Y_aug = sm.fit_resample(X, Y) # type:ignore

        
    return X_aug, Y_aug


def load_model_cohort(root: str, num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, random_state: int = 100, generate_syn_nonEZ: bool = True, node_num: str = "1"):  

    ## Load ModelCohort
    print(f"Loading ModelCohort for Node num: {node_num}")

    path = os.path.join(root,'Valid_NonEZvsEZ_ALL')

    RI_file = f"Valid_NonEZvsEZ_RI_node{node_num}_ALL.mat"
    Conn_file = f"Valid_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
    label_file = f"Valid_NonEZvsEZ_label_node{node_num}_ALL.mat"
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

    ## Load ValidCohort
    print(f"Loading ValidCohort for Node num: {node_num}")

    path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')

    RI_file = f"ValidCohort_NonEZvsEZ_RI_node{node_num}_ALL.mat"
    Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
    label_file = f"ValidCohort_NonEZvsEZ_label_node{node_num}_ALL.mat"
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

    X_combined_2 = X_combined_2[:14,:] # first 14 patients of the validation cohort

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat_aug = Y_mat_l[label_mat_name]
    Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

    Y_mat_aug_2 = Y_mat_aug_2[:14] # first 14 patients of the validation cohort

                
    # combine training node-level data
    X_train_node = np.concatenate((X_combined_1, X_combined_2), axis=0)

    Y_train_node = np.concatenate((Y_mat_aug_1, Y_mat_aug_2), axis=0)

    Y_train_node = Y_train_node.astype(int) # type:ignore
    
    print('Y_train_node: %s' % Counter(Y_train_node))

    # augment training data using SMOTE
    if np.sum(Y_train_node) == 1:
        X_train_node_aug, Y_train_node_aug = augment_1EZ_data(root, X_train_node, Y_train_node, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state, generate_syn_nonEZ=generate_syn_nonEZ, node_num=node_num)

    else:
        X_train_node_aug, Y_train_node_aug = augment_data(X_train_node, Y_train_node, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=random_state, generate_syn_nonEZ=generate_syn_nonEZ, node_num=node_num)

    Y_train_node_aug = Y_train_node_aug.astype(int) # type:ignore

    print('Y_train_node_augmented: %s' % Counter(Y_train_node_aug))    

    return X_train_node_aug, Y_train_node_aug, X_train_node, Y_train_node


def load_validation_cohort(root: str, node_num: str = "1"):    
      
    ## Load ValidCohort
    print(f"Loading ValidCohort for Node num: {node_num}")

    path = os.path.join(root,'ValidCohort_NonEZvsEZ_ALL')

    RI_file = f"ValidCohort_NonEZvsEZ_RI_node{node_num}_ALL.mat"
    Conn_file = f"ValidCohort_NonEZvsEZ_Conn_node{node_num}_ALL.mat"
    label_file = f"ValidCohort_NonEZvsEZ_label_node{node_num}_ALL.mat"
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

    X_combined_2 = X_combined_2[14:,:] # last 10 patients of the validation cohort

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat_aug = Y_mat_l[label_mat_name]
    Y_mat_aug_2 = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

    Y_mat_aug_2 = Y_mat_aug_2[14:] # last 10 patients of the validation cohort

    Y_mat_aug_2 = Y_mat_aug_2.astype(int) 

    print('Y_val_node_original: %s' % Counter(Y_mat_aug_2))

    return X_combined_2, Y_mat_aug_2


def save_aug_data_as_separate_nodes(save_dir: str, X: np.ndarray, Y: np.ndarray, mode: str = "train") -> None:
    
    for i in range(len(Y)):
        if mode == "train":
            savemat(os.path.join(save_dir,'X_train_aug_patient' + str(i) + '.mat'), {"X_aug_train_patient" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_train_aug_patient' + str(i) + '.mat'), {"Y_aug_train_patient" + str(i):Y[i]})

        elif mode == "validation":
            savemat(os.path.join(save_dir,'X_valid_orig_patient' + str(i) + '.mat'), {"X_orig_valid_patient" + str(i):X[i,:]})
            savemat(os.path.join(save_dir,'Y_valid_orig_patient' + str(i) + '.mat'), {"Y_orig_valid_patient" + str(i):Y[i]})

        else:
            raise KeyError(f"The mode must be either train or validation.")


def main(root: str, save_path_training: str, save_path_validation: str, num_samples_nonEZ: int = 50, num_samples_EZ: int = 50, generate_syn_nonEZ: bool = True): 

    node_numbers = []   

    num_nonEZs_train = []
    num_EZs_train = []

    num_nonEZs_train_orig = []
    num_EZs_train_orig = []

    num_nonEZs_val = []
    num_EZs_val = []

    for i in node_numbers_with_smote:

        node_numbers.append(i)

        ## Load and save the augmented training data per node
        X_train_aug, Y_train_aug, X_train_orig, Y_train_orig = load_model_cohort(root, num_samples_nonEZ=num_samples_nonEZ, num_samples_EZ=num_samples_EZ, random_state=100, generate_syn_nonEZ=generate_syn_nonEZ, node_num=i)  

        # calculate number of non_EZs and EZs for a given augmented training node
        num_nonEZs_train.append(len(Y_train_aug) - np.sum(Y_train_aug))
        num_EZs_train.append(np.sum(Y_train_aug))

        # calculate number of non_EZs and EZs for a given original training node
        num_nonEZs_train_orig.append(len(Y_train_orig) - np.sum(Y_train_orig))
        num_EZs_train_orig.append(np.sum(Y_train_orig))
       
        # save the augmented training data
        save_path_training = save_path_training

        save_dir_train_temp = 'Node_' + i
        save_dir_train_temp1 = 'Aug_Train_Data'
        save_dir_train = os.path.join(save_path_training, save_dir_train_temp, save_dir_train_temp1)

        if not os.path.exists(save_dir_train):
            os.makedirs(save_dir_train)
        
        save_aug_data_as_separate_nodes(save_dir_train, X_train_aug, Y_train_aug, mode="train")  # type:ignore   

        ## Load and save the original unaugmented validation data        
        X_val_orig, Y_val_orig = load_validation_cohort(root, node_num=i)  # type:ignore

        # calculate number of non_EZs and EZs for a given original validation node
        num_nonEZs_val.append(len(Y_val_orig) - np.sum(Y_val_orig))
        num_EZs_val.append(np.sum(Y_val_orig))

        # save the original validation data
        save_path_validation = save_path_validation

        save_dir_val_temp = 'Node_' + i
        save_dir_val_temp1 = 'Orig_Val_Data'        
        
        save_dir_val = os.path.join(save_path_validation, save_dir_val_temp, save_dir_val_temp1)
        if not os.path.exists(save_dir_val):
            os.makedirs(save_dir_val)

        save_aug_data_as_separate_nodes(save_dir_val, X_val_orig, Y_val_orig, mode="validation")  # type:ignore

    
    # dictionary of lists
    info_dict = {'Node #': node_numbers, 'NonEZ-train-orig': num_nonEZs_train_orig, 'EZ-train-orig': num_EZs_train_orig, 'NonEZ-train-aug': num_nonEZs_train, 'EZ-train-aug': num_EZs_train, 'NonEZ-val-orig': num_nonEZs_val, 'EZ-val-orig': num_EZs_val}    

    df = pd.DataFrame(info_dict)  

    # saving the dataframe
    path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/"  
    save_path = os.path.join(path, "Information")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename  = "info.xlsx"
    save_filepath = os.path.join(save_path, filename)

    df.to_excel(save_filepath, sheet_name='Sheet1', header=True, index=False)

    ################################################################################################################


if __name__ == "__main__":

    # Root Folder for the dataset
    # root='/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/'
    root='/home/share/Data/EZ_Pred_Dataset/All_Hemispheres/'

    save_path_training = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/'
    save_path_validation = '/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/'

    # save_path_training = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Temporal_Lobe/'
    # save_path_validation = '/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Temporal_Lobe/'

    # num_samples_nonEZ: Number of samples of non-EZ (class 0) to generate per node with SMOTE
    # num_samples_EZ: Number of samples of EZ (class 1) to generate per node with SMOTE
    
    main(root, save_path_training, save_path_validation, num_samples_nonEZ=60, num_samples_EZ=60, generate_syn_nonEZ=True)


