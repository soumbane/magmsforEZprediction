# MLP Model (2 branches) for Node-Level classification - both hemispheres
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

import xgboost
from xgboost import XGBClassifier as xgb

import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, random_split
from torch_geometric import seed_everything

from imblearn.over_sampling import SMOTE

# for reproducibility of algorithms
seed = 100 # (used for the most successful algorithm)

seed_everything(seed) # seed = 100 is used for all experiments

torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False  # type:ignore
torch.backends.cudnn.deterministic = True  # type:ignore


def calculate_metrics(y_pred, y_true, average_type='macro'):

    conf_mat = confusion_matrix(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average=average_type)

    acc = accuracy_score(y_true, y_pred)

    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec = precision_score(y_true, y_pred, average=average_type)

    rec = recall_score(y_true, y_pred, average=average_type)    

    # # auc = roc_auc_score(y_true, all_preds_for_auc, average=average_type, multi_class='ovo') # for multi-classes
    # auc = roc_auc_score(y_true, y_pred, average=average_type) # for binary classes

    # # target_names = ['class 0', 'class 1', 'class 2'] # for multi-classes
    # target_names = ['class 0', 'class 1'] # for binary classes
    # class_report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
    # return conf_mat, f1, acc, bal_acc, prec, rec, auc, class_report_dict
    return bal_acc, conf_mat

def print_metrics(bal_acc, conf_mat):

    print(f"Confusion Matrix: \n {conf_mat}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.show()  

    print(f"\nBalanced Accuracy: {bal_acc:.4f}")


# Root Folder
root='/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/'

def load_train_data(node_num: str):

    train_path = "DataSet/DataForSIMUL_2/Node_Level_Augmented_Datasets/All_Hemispheres/Train_NonEZvsEZ_ALL"

    train_RI_file = "Train_NonEZvsEZ_RI_node" + node_num + "_ALL.mat"
    train_Conn_file = "Train_NonEZvsEZ_Conn_node" + node_num + "_ALL.mat"
    train_label_file = "Train_NonEZvsEZ_label_node" + node_num + "_ALL.mat"

    raw_path_RI = os.path.join(root,train_path,train_RI_file)
    raw_path_label = os.path.join(root,train_path,train_label_file)
    raw_path_Conn = os.path.join(root,train_path,train_Conn_file)

    """Load the Data Matrix from .mat files.""" 
    X_mat_l = loadmat(raw_path_RI)
    X_mat_RI = X_mat_l['Augmented_RI']
    # print(f'RI Feature Matrix Shape: {X_mat_RI.shape}')

    """Load the Data Matrix from .mat files.""" 
    X_mat_lconn = loadmat(raw_path_Conn)
    X_mat_Conn = X_mat_lconn['Augmented_Conn']
    # print(f'Conn Feature Matrix Shape: {X_mat_Conn.shape}')

    # check for NaN values in X and replace NaN values with 0
    if (np.isnan(X_mat_RI).any()):  # type:ignore
        X_mat_RI = np.nan_to_num(X_mat_RI, nan=0)  

    if (np.isnan(X_mat_Conn).any()):  # type:ignore
        X_mat_Conn = np.nan_to_num(X_mat_Conn, nan=0)  

    X_mat_aug = np.concatenate((X_mat_RI,X_mat_Conn), axis=1)  # using both RI and Conn features
    # X_mat = X_mat_RI  # using only RI features
    # print(f'Final Augmented Feature Matrix Shape: {X_mat_aug.shape}')

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat_aug = Y_mat_l['Augmented_label']
    # print(f'GT-Labels shape:{Y_mat_aug.shape}')

    # Randomly shuffle X_train and Y_train with the same seed
    np.random.seed(0)
    np.random.shuffle(X_mat_aug)

    np.random.seed(0)
    np.random.shuffle(Y_mat_aug)
    Y_mat_aug = Y_mat_aug.reshape(Y_mat_aug.shape[0],)

    X_train = torch.from_numpy(X_mat_aug)
    Y_train = torch.from_numpy(Y_mat_aug).long() # for CrossEntropyLoss

    # print(np.bincount(Y_train))
    # print('X_train augmented shape: ', X_train.shape)
    # print('Y_train augmented shape: ', Y_train.shape)    

    return (X_train, Y_train)

def load_test_data(node_num: str):
    
    valid_path = "DataSet/DataForSIMUL_2/Node_Level_Augmented_Datasets/All_Hemispheres/Valid_NonEZvsEZ_ALL"

    valid_RI_file = "Valid_NonEZvsEZ_RI_node" + node_num + "_ALL.mat"
    valid_Conn_file = "Valid_NonEZvsEZ_Conn_node" + node_num + "_ALL.mat"
    valid_label_file = "Valid_NonEZvsEZ_label_node" + node_num + "_ALL.mat"

    raw_path_RI = os.path.join(root,valid_path,valid_RI_file)
    raw_path_label = os.path.join(root,valid_path,valid_label_file)
    raw_path_Conn = os.path.join(root,valid_path,valid_Conn_file)

    """Load the Data Matrix from .mat files.""" 
    X_mat_l = loadmat(raw_path_RI)
    X_mat_RI = X_mat_l['ModelCohort_NonEZvsEZ_RI']
    # print(f'Validation RI Feature Matrix Shape: {X_mat_RI.shape}')

    """Load the Data Matrix from .mat files.""" 
    X_mat_lconn = loadmat(raw_path_Conn)
    X_mat_Conn = X_mat_lconn['ModelCohort_NonEZvsEZ_Conn']
    # print(f'Validation Conn Feature Matrix Shape: {X_mat_Conn.shape}')

    # check for NaN values in X and replace NaN values with 0
    if (np.isnan(X_mat_RI).any()):  # type:ignore
        X_mat_RI = np.nan_to_num(X_mat_RI, nan=0)  

    if (np.isnan(X_mat_Conn).any()):  # type:ignore
        X_mat_Conn = np.nan_to_num(X_mat_Conn, nan=0)  

    X_mat = np.concatenate((X_mat_RI,X_mat_Conn), axis=1)  # using both RI and Conn features
    # X_mat = X_mat_RI  # using only RI features
    # print(f'Validation Final Feature Matrix Shape: {X_mat.shape}')

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat = Y_mat_l['ModelCohort_NonEZvsEZ_label']

    Y_mat_val = Y_mat.reshape(Y_mat.shape[0],)

    X_test = torch.from_numpy(X_mat)
    Y_test = torch.from_numpy(Y_mat_val).long() # for CrossEntropyLoss

    # print('X_test shape: ', X_test.shape)
    # print('Y_test shape: ', Y_test.shape)

    return (X_test, Y_test)

# Training and testing/validation dataloaders
def load_train_loader(X_train, Y_train):
   train_data = []
   for i in range(len(X_train)):
      train_data.append([X_train[i], Y_train[i]])

   train_loader = DataLoader(train_data, shuffle=True, batch_size=32)  # type:ignore
   return train_loader

def load_test_loader(X_test, Y_test):
   test_data = []
   for i in range(len(X_test)):
      test_data.append([X_test[i], Y_test[i]])
   test_loader = DataLoader(test_data, shuffle=False, batch_size=1)  # type:ignore
   return test_loader

# Define the model (This MLP model works for both hemispheres ONLY)
class MLP_ms(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers_RI = nn.Sequential(
      nn.Linear(1400, 1024),
      nn.Dropout(p=0.2),
      nn.ReLU()
    )

    self.layers_Conn = nn.Sequential(
      nn.Linear(499, 256),
      nn.Dropout(p=0.1),
      nn.ReLU()
    )

    self.layers_combined = nn.Sequential(
      nn.Linear(1280, 2)  # for CrossEntropyLoss
    )

  def forward(self, x):
    x_RI = self.layers_RI(x[:, :1400])
    x_Conn = self.layers_Conn(x[:,1400:])

    x_combined = torch.cat((x_RI,x_Conn), dim=1)

    x_final = self.layers_combined(x_combined)

    return x_final

def train_one_epoch(model, train_loader, optimizer, loss_fn, device=None):    
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for _, (x,y) in enumerate(tqdm(train_loader)):
        
        # Use GPU
        x.to(device) 

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info        
        pred = model(x)   
            
        # Calculating the loss and gradients
        loss = loss_fn(pred, y)

        loss.backward() 

        optimizer.step() 

        # Update tracking
        running_loss += loss.item()

        step += 1

        all_preds.append((pred.argmax(dim=-1)).cpu().detach().numpy()) # for CrossEntropyLoss
        all_labels.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    return running_loss/step

def test(model, test_loader, loss_fn, device=None):
    all_preds = []
    preds_for_auc = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for (x,y) in test_loader:

        x.to(device) 

        pred = model(x) 

        loss = loss_fn(pred, y)

         # Update tracking
        running_loss += loss.item()
        step += 1

        # preds_for_auc.append((F.softmax(pred, dim=1)).cpu().detach().numpy()) # for CrossEntropyLoss
        all_preds.append((pred.argmax(dim=-1)).cpu().detach().numpy())  # for CrossEntropyLoss
        all_labels.append(y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    # preds_for_auc = np.concatenate(preds_for_auc)
    all_labels = np.concatenate(all_labels).ravel()

    bal_acc, conf_mat = calculate_metrics(all_preds, all_labels, average_type='macro')

    return running_loss/step, bal_acc, conf_mat

def train_all_epochs_per_node(model, train_loader, test_loader, epochs=100, patience=10, 
save_dir='', exp_id='', loss_fn=None, optimizer=None, device=None):
    # Start training
    best_loss = float('inf')
    best_bal_acc = 0.0
    early_stopping_counter = 0
    best_epoch = 1
    all_best_epoch_nums = []

    for epoch in range(1, epochs+1): 
        if early_stopping_counter <= patience:

            # Training
            model.train()
            loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=device)
            print(f"Epoch {epoch} | Train Loss {loss:.4f}")

            # Validation
            model.eval()

            loss, bal_acc, conf_mat = test(model, test_loader, loss_fn, device=device)
            print(f"Epoch {epoch} | Test Loss {loss:.4f}")

            # Update best (highest) Balanced Accuracy
            if float(bal_acc) > best_bal_acc:

                print(f'Validation Balanced Accuracy increases from {best_bal_acc:.4f} to {bal_acc:.4f}')

                best_bal_acc = bal_acc
                
                print('Saving best model with highest Balanced Accuracy ...')
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, save_dir + "exp_node_" + str(exp_id) + "_best_model.pth")
                best_epoch = epoch
                all_best_epoch_nums.append(epoch)

                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        else:
            print("Early stopping due to no improvement.")
        
            print(f"Finishing training with highest Balanced Accuracy: {best_bal_acc}")
            break


    # Final Test with trained model
    print('\nTesting with the best model with highest Balanced Accuracy on test set ...\n')
    model.load_state_dict(torch.load(save_dir + "exp_node_" + str(exp_id) + "_best_model.pth"))

    loss, bal_acc, conf_mat = test(model, test_loader, loss_fn)

    print("Printing Test Set Evaluation Metrics ....\n")

    print_metrics(bal_acc, conf_mat)

    print(f"\nBest scores occur at {best_epoch} epoch number")

    return bal_acc, conf_mat

def get_list_of_node_nums():
    node_numbers_with_smote = [
        "None","1","2","3","5","6","11","12","13","14","17","18","19","20",
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
        "600","601","602","603","604","605","606","607","608","609","610","611","612","613","614","615","616","617",
        "618","619","620","621","622","623","624","625","626","627","628","629","630","631","632","633","634","635",
        "636","637","638","639","640","641","642","643","644","645","646","647","648","649","650","651",
        "652","653","655","656","657","658","659","660","661","662","663","664","665","666","667","668",
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

save_dir = "./checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main loop to train and test the model over all the nodes
total_nodes = 998

# choose model
model_name = 'MLP_ms'

# Define the model
if model_name == 'MLP_ms':
    # Initialize the model
    model = MLP_ms()
    model.double()
else:
    raise NotImplementedError("Unknown Model.")

# Train and test the model with these settings
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # type: ignore

epochs = 100
patience = 20 # needed for early stopping

# Define training loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00005, weight_decay=5e-4)

count = 0

node_num_list = [] # stores the node numbers for all nodes
bal_acc_list = [] # stores the balanced accuracies for all nodes
conf_mat_list = [] # stores the confusion matrices for all nodes

# list of all 827 nodes for which SMOTE is possible (atleast 1 EZ)
node_numbers_with_smote = get_list_of_node_nums()

for i in range(1,total_nodes+1):
    if (str(i)+"V3" in node_numbers_with_smote or str(i)+"V2" in node_numbers_with_smote 
    or str(i) in node_numbers_with_smote):        
        print(f'Node Num: {i}')
        print(f'count: {count}')

        if count == 0:
            node_num = node_numbers_with_smote[i]
        else:
            node_num = node_numbers_with_smote[i-count]

        exp_id = node_num # Experiment ID

        print(f'Node num as string: {node_num}')

        # load the data for the given node
        X_train, Y_train = load_train_data(node_num)
        train_loader = load_train_loader(X_train, Y_train)

        X_test, Y_test = load_test_data(node_num)
        test_loader = load_test_loader(X_test, Y_test)

        # Train the model
        print(f'Training and Evaluating on Node number: {i}')           

        # Evaluate Trained Model with evaluation metrics
        bal_acc, conf_mat = train_all_epochs_per_node(model, train_loader, test_loader, epochs=epochs, 
        patience=patience, save_dir=save_dir, exp_id=exp_id, loss_fn=loss_fn, optimizer=optimizer, device=device)
        # print(bal_acc)
        # print(conf_mat)

        # for saving balanced accuracy and confusion matrix of nodes with SMOTE in a csv file
        node_num_list.append(i)
        bal_acc_list.append(bal_acc)
        conf_mat_list.append(conf_mat)

    elif (str(i)+"V3" not in node_numbers_with_smote or str(i)+"V2" not in node_numbers_with_smote 
    or str(i) not in node_numbers_with_smote):
        print(f'Node Num: {i}')
        count += 1
        # for saving balanced accuracy and confusion matrix as 0 of nodes with 0 EZ in a csv file
        node_num_list.append(i) # node_num with 0 EZ patients
        bal_acc_list.append(0)
        conf_mat_list.append(0)

    elif i > len(node_numbers_with_smote):
        print(f'Node Num: {i}')
        # for saving balanced accuracy and confusion matrix as 0 of nodes with 0 EZ in a csv file
        node_num_list.append(i) # node_num with 0 EZ patients
        bal_acc_list.append(0)
        conf_mat_list.append(0)

# dictionary of lists
dict = {'node number': node_num_list, 'balanced_acc': bal_acc_list, 'confusion matrix': conf_mat_list}
    
df = pd.DataFrame(dict)

# saving the dataframe
metric_filename = model_name + '_metrics_test_1.csv'
save_metrics_path = os.path.join(root,"metrics",metric_filename)
df.to_csv(save_metrics_path, header=False, index=False)

###################################################################################################################
# The following is for evaluation on the validation set
# for reproducibility of algorithms
seed = seed # (used for the most successful algorithm)

seed_everything(seed) # seed = 100 is used for all experiments

torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False  # type:ignore
torch.backends.cudnn.deterministic = True  # type:ignore


def load_valid_data(node_num: str):
    
    valid_path = "DataSet/DataForSIMUL_2/Node_Level_Augmented_Datasets/All_Hemispheres/ValidCohort_NonEZvsEZ_ALL"

    valid_RI_file = "ValidCohort_NonEZvsEZ_RI_node" + node_num + "_ALL.mat"
    valid_Conn_file = "ValidCohort_NonEZvsEZ_Conn_node" + node_num + "_ALL.mat"
    valid_label_file = "ValidCohort_NonEZvsEZ_label_node" + node_num + "_ALL.mat"

    raw_path_RI = os.path.join(root,valid_path,valid_RI_file)
    raw_path_label = os.path.join(root,valid_path,valid_label_file)
    raw_path_Conn = os.path.join(root,valid_path,valid_Conn_file)

    """Load the Data Matrix from .mat files.""" 
    X_mat_l = loadmat(raw_path_RI)
    X_mat_RI = X_mat_l['ValidCohort_NonEZvsEZ_RI']
    # print(f'Validation RI Feature Matrix Shape: {X_mat_RI.shape}')

    """Load the Data Matrix from .mat files.""" 
    X_mat_lconn = loadmat(raw_path_Conn)
    X_mat_Conn = X_mat_lconn['ValidCohort_NonEZvsEZ_Conn']
    # print(f'Validation Conn Feature Matrix Shape: {X_mat_Conn.shape}')

    # check for NaN values in X and replace NaN values with 0
    if (np.isnan(X_mat_RI).any()):  # type:ignore
        X_mat_RI = np.nan_to_num(X_mat_RI, nan=0)  

    if (np.isnan(X_mat_Conn).any()):  # type:ignore
        X_mat_Conn = np.nan_to_num(X_mat_Conn, nan=0)  

    X_mat = np.concatenate((X_mat_RI,X_mat_Conn), axis=1)  # using both RI and Conn features
    # X_mat = X_mat_RI  # using only RI features
    # print(f'Validation Final Feature Matrix Shape: {X_mat.shape}')

    """Load the Label Matrix from .mat files.""" 
    Y_mat_l = loadmat(raw_path_label)
    Y_mat = Y_mat_l['ValidCohort_NonEZvsEZ_label']

    Y_mat_val = Y_mat.reshape(Y_mat.shape[0],)

    X_val = torch.from_numpy(X_mat)
    Y_val = torch.from_numpy(Y_mat_val).long() # for CrossEntropyLoss

    # print('X_val shape: ', X_val.shape)
    # print('Y_val shape: ', Y_val.shape)

    return (X_val, Y_val)

# Load Validation loader
def load_valid_loader(X_val, Y_val):
   valid_data = []
   for i in range(len(X_val)):
      valid_data.append([X_val[i], Y_val[i]])

   valid_loader = DataLoader(valid_data, shuffle=False, batch_size=1)  # type:ignore
   return valid_loader


save_dir = "./checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main loop to validate the trained model over all the nodes
total_nodes = 998

# choose model
model_name = 'MLP_ms'

# Define the model
if model_name == 'MLP_ms':
    # Initialize the model
    model = MLP_ms()
    model.double()
else:
    raise NotImplementedError("Unknown Model.")

# Validate the trained the model with these settings
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # type: ignore

# Define training loss function for validation loss
loss_fn = torch.nn.CrossEntropyLoss()

count = 0

node_num_list = [] # stores the node numbers for all nodes
bal_acc_list = [] # stores the balanced accuracies for all nodes
conf_mat_list = [] # stores the confusion matrices for all nodes

# list of all 827 nodes for which SMOTE is possible (atleast 1 EZ)
node_numbers_with_smote = get_list_of_node_nums()

for i in range(1,total_nodes+1):
    if (str(i)+"V3" in node_numbers_with_smote or str(i)+"V2" in node_numbers_with_smote 
    or str(i) in node_numbers_with_smote):        
        print(f'Node Num: {i}')

        if count == 0:
            node_num = node_numbers_with_smote[i]
        else:
            node_num = node_numbers_with_smote[i-count]

        exp_id = node_num # Experiment ID

        print(f'Node num as string: {node_num}')

        # load the data for the given node
        X_val, Y_val = load_valid_data(node_num)
        valid_loader = load_valid_loader(X_val, Y_val)

        # Train the model
        print(f'Validating on Node number: {i}')           

        # Validate Trained Model with evaluation metrics
        print('\nTesting with the best model with highest Balanced Accuracy on validation set ...\n\n')
        model.load_state_dict(torch.load(save_dir + "exp_node_" + str(exp_id) + "_best_model.pth"))

        _, bal_acc, conf_mat = test(model, valid_loader, loss_fn, device=device)

        print("Printing Evaluation Metrics ....\n\n")

        # print_metrics(bal_acc, conf_mat)

        # for saving balanced accuracy and confusion matrix of nodes with SMOTE in a csv file
        node_num_list.append(i)
        bal_acc_list.append(bal_acc)
        conf_mat_list.append(conf_mat)

    elif (str(i)+"V3" not in node_numbers_with_smote or str(i)+"V2" not in node_numbers_with_smote 
    or str(i) not in node_numbers_with_smote):
        print(f'Node Num: {i}')
        count += 1
        # for saving balanced accuracy and confusion matrix as 0 of nodes with 0 EZ in a csv file
        node_num_list.append(i) # node_num with 0 EZ patients
        bal_acc_list.append(0)
        conf_mat_list.append(0)

    elif i > len(node_numbers_with_smote):
        print(f'Node Num: {i}')
        # for saving balanced accuracy and confusion matrix as 0 of nodes with 0 EZ in a csv file
        node_num_list.append(i) # node_num with 0 EZ patients
        bal_acc_list.append(0)
        conf_mat_list.append(0)

# dictionary of lists
dict = {'node number': node_num_list, 'balanced_acc': bal_acc_list, 'confusion matrix': conf_mat_list}
    
df = pd.DataFrame(dict)

# saving the dataframe
metric_filename = model_name + '_metrics_validation_1.csv'
save_metrics_path = os.path.join(root,"metrics",metric_filename)
df.to_csv(save_metrics_path, header=False, index=False)

print(np.mean(bal_acc_list))
print(bal_acc_list)

# calculate mean of non-zero elements: nodes with atleast 1 EZ in model cohort
bal_acc_list_non_zero = [bal_acc_list[i] for i in range(len(bal_acc_list)) if bal_acc_list[i] != 0]

# bal_acc_list_non_zero.append(1) # for SVM Only for node 378
print(len(bal_acc_list_non_zero))
print(np.mean(bal_acc_list_non_zero))

