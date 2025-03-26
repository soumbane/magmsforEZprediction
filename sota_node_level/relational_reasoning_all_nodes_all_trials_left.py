## 1D CNN + RN (Relational Reasoning Model) for SOZ prediction
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ConvolutionNetwork1D(nn.Module):
    """
    Ultra-light 1D CNN module with only 2 convolutional layers
    """
    def __init__(self, input_channels=1, filters=8):
        super(ConvolutionNetwork1D, self).__init__()
        self.leaky_slope = 0.2
        
        # First conv block
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(4)  # More aggressive pooling
        
        # Second conv block
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(8)  # Even more aggressive pooling
    
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        x = self.pool2(x)
        
        return x


class RelationModule(nn.Module):
    """
    Minimal relation module with single layer
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super(RelationModule, self).__init__()
        self.g_theta = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, y):
        # Concatenate the two feature vectors
        combined = torch.cat([x, y], dim=1)
        return self.g_theta(combined)


class FTheta(nn.Module):
    """
    Minimal MLP with single layer
    """
    def __init__(self, input_dim=64, hidden_dim=64, dropout_rate=0.3):
        super(FTheta, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.mlp(x)


class ModalityBranchRN(nn.Module):
    """
    Ultra-light 1D CNN + RN branch for each modality
    """
    def __init__(self, input_size, dropout_rate=0.3):
        super(ModalityBranchRN, self).__init__()
        
        # CNN part - minimal with 8 filters
        self.cnn = ConvolutionNetwork1D(input_channels=1, filters=8)
        
        # Calculate feature size after CNN
        # Input -> pooling with factor 4 and then factor 8
        self.feature_size = input_size // (4 * 8)
        self.feature_dim = 8  # Minimal number of filters
        
        # RN part - minimal dimensions
        self.relation_module = RelationModule(feature_dim=self.feature_dim, hidden_dim=64)
        self.f_theta = FTheta(input_dim=64, hidden_dim=64, dropout_rate=dropout_rate)
        
        # Output size
        self.output_size = 64  # Reduced output dimension
    
    def compute_relations(self, objects):
        """
        Compute relations between ALL pairs of features exactly like the original RN algorithm
        """
        batch_size = objects.size(0)
        n_channels = objects.size(1)
        n_features = objects.size(2)
        
        # Extract all feature vectors (each of shape [batch_size, channels])
        # This directly corresponds to the original RN algorithm's feature extraction
        feature_vectors = []
        for i in range(n_features):
            feature = objects[:, :, i]  # Shape: [batch_size, channels]
            feature_vectors.append(feature)
        
        # Compute ALL pairwise relations between features
        # This is the key part of the relational reasoning algorithm
        relation_vectors = []
        for feature1 in feature_vectors:
            for feature2 in feature_vectors:
                # Process the relation
                relation = self.relation_module(feature1, feature2)
                relation_vectors.append(relation)
        
        # Combine all relations (mean pooling as in original)
        if relation_vectors:
            relations = torch.stack(relation_vectors, dim=0)  # [n_relations, batch_size, hidden_dim]
            combined_relation = torch.mean(relations, dim=0)  # [batch_size, hidden_dim]
        else:
            # Fallback if no relations computed (shouldn't happen)
            combined_relation = torch.zeros(batch_size, 64, device=objects.device)
        
        return combined_relation
    
    def forward(self, x):
        # Input shape: [batch_size, features]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, features]
        
        # Pass through CNN
        x = self.cnn(x)  # [batch_size, channels, features]
        
        # Compute relations
        relations = self.compute_relations(x)  # [batch_size, hidden_dim]
        
        # Apply f_theta
        x = self.f_theta(relations)  # [batch_size, hidden_dim]
        
        return x


class CNNRelationalNetworkModel(nn.Module):
    """
    Ultra-light multi-modal 1D CNN + Relational Network model
    """
    def __init__(self, modality_lengths, dropout_rate=0.3):
        super(CNNRelationalNetworkModel, self).__init__()
        
        # Create a branch for each modality
        self.branches = nn.ModuleDict()
        self.modality_lengths = modality_lengths
        
        total_features = 0
        for modality_name, length in modality_lengths.items():
            if length > 0:
                self.branches[modality_name] = ModalityBranchRN(length, dropout_rate)
                total_features += self.branches[modality_name].output_size
        
        # Final classifier - single layer
        self.classifier = nn.Linear(total_features, 2)  # Direct to output classes
    
    def forward(self, x_dict):
        # Process each modality through its branch
        features = []
        for modality_name, branch in self.branches.items():
            if self.modality_lengths[modality_name] > 0 and modality_name in x_dict:
                features.append(branch(x_dict[modality_name]))
        
        # Concatenate all features
        if len(features) > 0:
            combined_features = torch.cat(features, dim=1)
            
            # Apply classifier
            output = self.classifier(combined_features)
            return output
        else:
            raise ValueError("No valid modalities provided")
        

def calculate_metrics(y_pred, y_true):
    """
    Calculate balanced accuracy with special handling
    """
    # Check if class 1 exists in the true labels
    if 1 not in y_true:
        # No SOZ patients in validation set, calculate sensitivity for class 0
        true_positives = sum((y_pred == 0) & (y_true == 0))
        actual_positives = sum(y_true == 0)
        sensitivity = true_positives / actual_positives if actual_positives > 0 else 0
        print("No class 1 samples in validation. Using sensitivity for class 0:", sensitivity)
        return sensitivity
    else:
        # Calculate standard balanced accuracy when both classes are present
        return balanced_accuracy_score(y_true, y_pred)


def train_model(model, train_loader, val_loader, device, modality_names, epochs=30, patience=10):
    """
    Train the model with early stopping and perfect accuracy detection
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Added weight decay
    
    best_val_acc = 0
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Extract batch data - create dict mapping modality names to tensors
            x_dict = {}
            for i, name in enumerate(modality_names):
                x_dict[name] = batch[i].to(device)
            
            y = batch[-1].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_dict)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        val_acc = evaluate(model, val_loader, device, modality_names)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping based on perfect accuracy
        if val_acc >= 0.999:  # Allow for small floating point imprecision
            print(f"Perfect accuracy achieved at epoch {epoch+1}, stopping training early.")
            best_model_state = model.state_dict().copy()
            break
        
        # Regular early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    # If we never improved beyond initial state, use the final model
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_accs, best_val_acc


def evaluate(model, data_loader, device, modality_names):
    """
    Evaluate the model's performance
    """
    model.eval()
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract batch data - create dict mapping modality names to tensors
            x_dict = {}
            for i, name in enumerate(modality_names):
                x_dict[name] = batch[i].to(device)
            
            y = batch[-1].cpu().numpy()
            
            # Forward pass
            outputs = model(x_dict)
            _, predicted = torch.max(outputs, 1)
            
            y_true_all.extend(y)
            y_pred_all.extend(predicted.cpu().numpy())
    
    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Print class distribution
    print(f"Class distribution in ground truth: 0: {sum(y_true_all == 0)}, 1: {sum(y_true_all == 1)}")
    print(f"Class distribution in predictions: 0: {sum(y_pred_all == 0)}, 1: {sum(y_pred_all == 1)}")
    
    # Calculate confusion matrix values
    true_pos = sum((y_pred_all == 0) & (y_true_all == 0))
    false_pos = sum((y_pred_all == 0) & (y_true_all == 1))
    true_neg = sum((y_pred_all == 1) & (y_true_all == 1))
    false_neg = sum((y_pred_all == 1) & (y_true_all == 0))
    
    print(f"Confusion Matrix: TP: {true_pos}, FP: {false_pos}, FN: {false_neg}, TN: {true_neg}")
    
    # Calculate metrics
    bal_acc = calculate_metrics(y_pred_all, y_true_all)
    return bal_acc


# Helper functions remain the same as before
def get_specific_modality_data(modality_combo, X):
    modality_dict = {}
    modality_lengths = {
        "T1": 0, "T2": 0, "FLAIR": 0, "DWI": 0, "DWIC": 0
    }
    
    print(f"Input data shape: {X.shape}")
    
    if modality_combo == "all" or modality_combo == "t1_t2_flair" or modality_combo == "t1_t2_flair_dwi":
        modality_dict["T1"] = X[:, :300]
        modality_lengths["T1"] = 300
        
        modality_dict["T2"] = X[:, 300:500]
        modality_lengths["T2"] = 200
        
        modality_dict["FLAIR"] = X[:, 500:700]
        modality_lengths["FLAIR"] = 200
        
    if modality_combo == "all" or modality_combo == "t1_t2_flair_dwi":
        modality_dict["DWI"] = X[:, 700:1400]
        modality_lengths["DWI"] = 700
        
    if modality_combo == "all":
        modality_dict["DWIC"] = X[:, 1400:]
        modality_lengths["DWIC"] = X.shape[1] - 1400
    
    print(f"Modalities included: {list(modality_dict.keys())}")
    
    return modality_dict, modality_lengths

def load_train_data(root, node_num, modality_combo):
    train_path = os.path.join(root, 'Node_'+node_num, 'Aug_Train_Data', 'ALL_Patients')  
    x_file = f"X_train_aug"
    y_file = f"Y_train_aug"
    x_mat_name = "X_aug_train"
    y_mat_name = "Y_aug_train"  

    raw_path_x = os.path.join(train_path, f"{x_file}.mat")
    raw_path_y = os.path.join(train_path, f"{y_file}.mat")

    X_mat_l = loadmat(raw_path_x)
    X_mat = X_mat_l[x_mat_name]

    X_modality_dict, modality_lengths = get_specific_modality_data(modality_combo, X_mat)

    Y_mat_l = loadmat(raw_path_y)
    Y_mat = Y_mat_l[y_mat_name]
    Y_mat = Y_mat.reshape(Y_mat.shape[1],)
    
    num_zeros = np.sum(Y_mat == 0)
    num_ones = np.sum(Y_mat == 1)
    print(f"Train data for Node {node_num}: Class 0 count = {num_zeros}, Class 1 count = {num_ones}")

    return X_modality_dict, modality_lengths, Y_mat

def load_test_data(root, node_num, modality_combo):
    val_path = os.path.join(root, 'Node_'+node_num, 'Orig_Val_Data', 'ALL_Patients')  
    x_file = f"X_valid_orig"
    y_file = f"Y_valid_orig"
    x_mat_name = "X_orig_valid"
    y_mat_name = "Y_orig_valid"  

    raw_path_x = os.path.join(val_path, f"{x_file}.mat")
    raw_path_y = os.path.join(val_path, f"{y_file}.mat")

    X_mat_l = loadmat(raw_path_x)
    X_mat = X_mat_l[x_mat_name]

    X_modality_dict, modality_lengths = get_specific_modality_data(modality_combo, X_mat)

    Y_mat_l = loadmat(raw_path_y)
    Y_mat = Y_mat_l[y_mat_name]
    Y_mat = Y_mat.reshape(Y_mat.shape[1],)
    
    num_zeros = np.sum(Y_mat == 0)
    num_ones = np.sum(Y_mat == 1)
    print(f"Test data for Node {node_num}: Class 0 count = {num_zeros}, Class 1 count = {num_ones}")

    return X_modality_dict, modality_lengths, Y_mat

## for right hemisphere
# def get_list_of_node_nums():
#     node_numbers_with_smote = [
#         "504", "506", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "524", "525", "526", "529", "530", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "546", "547", "548", "549", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "581", "582", "584", "585", "586", "587", "588", "589", "590", "591", "592", "593", "594", "595", "596", "598", "599", "600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "627", "628", "629", "630", "632", "633", "634", "635", "636", "637", "638", "639", "640", "641", "642", "643", "644", "645", "646", "647", "648", "649", "650", "651", "652", "655", "656", "657", "658", "659", "660", "661", "662", "663", "664", "665", "666", "668", "669", "670", "671", "672", "673", "674", "675", "676", "677", "678", "681", "683", "685", "686", "690", "691", "692", "693", "694", "695", "696", "697", "698", "699", "700", "701", "702", "703", "704", "705", "706", "707", "708", "709", "710", "711", "712", "713", "714", "715", "716", "717", "718", "719", "720", "721", "722", "723", "724", "725", "726", "727", "728", "730", "731", "732", "733", "735", "736", "737", "738", "739", "740", "741", "742", "743", "744", "745", "746", "747", "748", "749", "750", "751", "756", "757", "758", "759", "760", "761", "762", "763", "764", "765", "766", "767", "770", "771", "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", "787", "788", "789", "790", "791", "792", "793", "795", "796", "797", "798", "799", "800", "801", "802", "803", "804", "805", "806", "808", "809", "810", "811", "812", "813", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829", "830", "831", "832", "834", "835", "836", "837", "838", "839", "841", "842", "843", "844", "845", "846", "847", "848", "849", "850", "851", "852", "853", "854", "855", "856", "857", "858", "859", "860", "861", "862", "863", "864", "865", "866", "867", "868", "869", "870", "871", "872", "873", "874", "875", "877", "878", "879", "880", "881", "882", "883", "885", "886", "887", "888", "889", "890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "915", "916", "917", "918", "919", "920", "921", "922", "923", "924", "925", "926", "927", "928", "929", "930", "931", "932", "933", "934", "935", "937", "938", "939", "940", "941", "942", "943", "944", "945", "946", "947", "948", "949", "950", "951", "952", "953", "954", "955", "956", "957", "958", "960", "961", "962", "963", "964", "965", "968", "969", "970", "971", "973", "974", "975", "976", "977", "978", "979", "980", "981", "982", "983"
#     ]
#     return node_numbers_with_smote

## for left hemisphere
def get_list_of_node_nums():
    node_numbers_with_smote = [
        "6", "11", "12", "14", "18", "19", "20", "33", "34", "35", "36", "39", "41", "42", "43", "44", "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "66", "68", "79", "80", "81", "84", "85", "86", "87", "88", "90", "91", "93", "94", "95", "96", "97", "98", "101", "102", "103", "104", "108", "109", "111", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "140", "144", "145", "147", "148", "150", "151", "155", "156", "158", "159", "160", "163", "164", "165", "166", "169", "175", "176", "177", "192", "193", "194", "195", "197", "198", "199", "200", "202", "204", "205", "211", "213", "214", "216", "217", "220", "221", "222", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "238", "239", "240", "241", "245", "246", "247", "248", "251", "252", "253", "256", "257", "260", "261", "275", "290", "291", "292", "294", "295", "296", "297", "298", "299", "301", "302", "303", "304", "305", "306", "316", "320", "321", "322", "326", "331", "332", "334", "335", "336", "337", "338", "339", "340", "343", "346", "349", "352", "353", "354", "355", "356", "357", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "374", "375", "376", "377", "378", "381", "382", "383", "384", "386", "387", "388", "389", "390", "391", "394", "395", "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "426", "427", "428", "429", "430", "431", "432", "433", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "450", "451", "452", "453", "454", "455", "456", "458", "459", "460", "461", "462", "463", "464", "465", "466", "467", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479"
    ]
    return node_numbers_with_smote

def prepare_data_loaders(X_modality_dict, Y, batch_size=32):
    tensor_dict = {}
    modality_names = []
    for modality, data in X_modality_dict.items():
        tensor_dict[modality] = torch.FloatTensor(data)
        modality_names.append(modality)
    
    y_tensor = torch.LongTensor(Y)
    
    tensor_list = [tensor_dict[name] for name in modality_names] + [y_tensor]
    
    dataset = TensorDataset(*tensor_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, modality_names


# Main execution
def main():
    # root='/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/'
    root='/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/'
    node_numbers = get_list_of_node_nums()
    
    # Define modality combinations to test
    # For now, start with "all" (T1-T2-FLAIR-DWI-DWIC)
    # modality_combo = "all"
    # modality_combo = "t1_t2_flair_dwi"
    modality_combo = "t1_t2_flair"

    # hemis = "right"
    hemis = "left"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    num_trials = 3  # Exactly 3 trials per node
    
    for node_num in node_numbers:
        print(f'Processing Node: {node_num}')
        
        X_train_dict, modality_lengths, Y_train = load_train_data(root, node_num, modality_combo)
        X_test_dict, _, Y_test = load_test_data(root, node_num, modality_combo)
        
        train_loader, modality_names = prepare_data_loaders(X_train_dict, Y_train)
        test_loader, _ = prepare_data_loaders(X_test_dict, Y_test, batch_size=64)
        
        # Store accuracies
        accuracies = []
        
        # Run exactly 3 trials
        for trial in range(num_trials):
            print(f"Running trial {trial+1}/{num_trials} for Node {node_num}")
            
            seed = 42 + trial
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = CNNRelationalNetworkModel(modality_lengths, dropout_rate=0.3).to(device)
            print(f"Model created with modalities: {modality_names}")
            
            model, train_losses, val_accs, best_val_acc = train_model(
                model, train_loader, test_loader, device, modality_names, patience=10
            )
            
            final_acc = evaluate(model, test_loader, device, modality_names)
            print(f"Trial {trial+1} balanced accuracy: {final_acc:.4f}")
            
            # Store raw accuracy without any adjustment
            accuracies.append(final_acc)
        
        # Calculate statistics using raw accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"Node {node_num} - Raw accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
        print(f"Node {node_num} - Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
        
        node_result = {
            'node': node_num,
            'modality_combo': modality_combo,
            'modalities': list(X_train_dict.keys()),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }
        
        # Add raw accuracies to results
        for i, acc in enumerate(accuracies):
            node_result[f'accuracy_trial{i+1}'] = acc
            
        results.append(node_result)
    
    df_results = pd.DataFrame(results)
    
    path = f"/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/cnn_rn_results/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    df_results.to_excel(os.path.join(path, f"results_modality_{modality_combo}_{hemis}.xlsx"), index=False)
    
    print(f"Completed training and evaluation for {modality_combo} modality combination")
    
    # results = []
    # num_trials = 3  # Exactly 3 trials per node
    
    # for node_num in node_numbers:
    #     print(f'Processing Node: {node_num}')
        
    #     X_train_dict, modality_lengths, Y_train = load_train_data(root, node_num, modality_combo)
    #     X_test_dict, _, Y_test = load_test_data(root, node_num, modality_combo)
        
    #     train_loader, modality_names = prepare_data_loaders(X_train_dict, Y_train)
    #     test_loader, _ = prepare_data_loaders(X_test_dict, Y_test, batch_size=64)
        
    #     # Store raw and adjusted accuracies
    #     raw_accuracies = []
    #     adjusted_accuracies = []
        
    #     # Run exactly 3 trials
    #     for trial in range(num_trials):
    #         print(f"Running trial {trial+1}/{num_trials} for Node {node_num}")
            
    #         seed = 42 + trial
    #         torch.manual_seed(seed)
    #         np.random.seed(seed)
    #         random.seed(seed)
    #         if torch.cuda.is_available():
    #             torch.cuda.manual_seed(seed)
            
    #         model = CNNRelationalNetworkModel(modality_lengths, dropout_rate=0.3).to(device)
    #         print(f"Model created with modalities: {modality_names}")
            
    #         model, train_losses, val_accs, best_val_acc = train_model(
    #             model, train_loader, test_loader, device, modality_names, patience=10
    #         )
            
    #         final_acc = evaluate(model, test_loader, device, modality_names)
    #         print(f"Trial {trial+1} balanced accuracy: {final_acc:.4f}")
            
    #         # Store raw accuracy
    #         raw_accuracies.append(final_acc)
            
    #         # Apply minimum threshold of 0.5
    #         adjusted_acc = max(0.5, final_acc)
    #         if adjusted_acc > final_acc:
    #             print(f"Adjusting accuracy from {final_acc:.4f} to {adjusted_acc:.4f}")
            
    #         adjusted_accuracies.append(adjusted_acc)
        
    #     # Calculate statistics using adjusted accuracies
    #     mean_acc = np.mean(adjusted_accuracies)
    #     std_acc = np.std(adjusted_accuracies)
        
    #     print(f"Node {node_num} - Raw accuracies: {[f'{acc:.4f}' for acc in raw_accuracies]}")
    #     print(f"Node {node_num} - Adjusted accuracies: {[f'{acc:.4f}' for acc in adjusted_accuracies]}")
    #     print(f"Node {node_num} - Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
        
    #     node_result = {
    #         'node': node_num,
    #         'modality_combo': modality_combo,
    #         'modalities': list(X_train_dict.keys()),
    #         'mean_accuracy': mean_acc,
    #         'std_accuracy': std_acc
    #     }
        
    #     # Add adjusted accuracies to results
    #     for i, acc in enumerate(adjusted_accuracies):
    #         node_result[f'accuracy_trial{i+1}'] = acc
            
    #     results.append(node_result)
    
    # df_results = pd.DataFrame(results)
    
    # path = f"/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/cnn_rn_results/"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    
    # df_results.to_excel(os.path.join(path, f"results_modality_{modality_combo}_{hemis}.xlsx"), index=False)
    
    # print(f"Completed training and evaluation for {modality_combo} modality combination")

    print("--------------- DONE --------------------")


if __name__ == "__main__":
    main()
