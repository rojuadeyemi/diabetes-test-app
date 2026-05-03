from utility.utility_functions import plot_confusion_matrix,plot_roc_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                        f1_score, roc_auc_score,classification_report)

def evaluate_performance(pipeline,model_name, X_test, y_test):

    y_test = y_test.values.ravel().astype('int')
    y_pred_test = pipeline.predict(X_test)
    y_pred_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Obtain performance metrics of the model 
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_pred_prob_test)

    print(f"{model_name} Performance")
    print(f"Accuracy: {accuracy_test:.2f}")
    print(f"Precision: {precision_test:.2f}")
    print(f"Recall: {recall_test:.2f}")
    print(f"F1 Score: {f1_test:.2f}")
    print(f"ROC AUC: {roc_auc_test:.2f}")
    print("\n")

    # Plot ROC Curve
    plot_roc_curve(pipeline, X_test, y_test, model_name)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_test, model_name)

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred_test))