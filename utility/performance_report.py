from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score
import os
from utility.utility_functions import plot_roc_curve, plot_confusion_matrix
from utility.model_evaluation import load_model_and_data
import pandas as pd

def comprehensive_model_report(model_path, X_test_path, y_test_path, model_name):
    model, X_test, y_test = load_model_and_data(model_path, X_test_path, y_test_path)
    y_pred = model.predict(X_test)

    # ROC Curve AUC
    roc_auc = plot_roc_curve(model, X_test, y_test, "Best "+model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, "Best "+model_name)

    # Classification Report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Model Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'roc_auc': [roc_auc]
    }
    path ="./report"
    os.makedirs(path,exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(path, "model_performance_report.csv"),index=False)
    pd.DataFrame(class_report).to_csv(os.path.join(path, "model_classification_report.csv"))
    print(f"Kindly check '{path}' on this machine for the reports")

def find_best(folder_path):
    
    files = os.listdir(folder_path)
    
    # Filter file that start with the specified "cv_"
    model_file = [f for f in files if f.startswith("cv_")]
    
    # return model path
    
    return model_file[0], os.path.join(folder_path, model_file[0])

def main(folder_path,X_test_path, y_test_path):

    model_name,model_path = find_best(folder_path)
    print(model_name)
    comprehensive_model_report(model_path, X_test_path, y_test_path, model_name)

if __name__ == "__main__":

    folder_path = "./model"
    X_test_path = "./processed_data/X_test.csv"
    y_test_path = "./processed_data/y_test.csv"
    main(folder_path,X_test_path,y_test_path)
