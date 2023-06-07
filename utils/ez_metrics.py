# Metrics to calculate the performance of MAG-MS EZ Prediction model

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def calculate_metrics(y_pred, y_true):

    conf_mat = confusion_matrix(y_true, y_pred)

    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return bal_acc, conf_mat

def print_metrics(bal_acc, conf_mat):

    print(f"Confusion Matrix: \n {conf_mat}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.show()  

    print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    