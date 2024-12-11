import pandas as pd
from sklearn.metrics import  classification_report
from utility.utility_functions import load_model,plot_confusion_matrix,plot_roc_curve,load_model_and_data

# Then evaluate the model
def evaluate_model(model_path, X_test_path, y_test_path, model_name):
    model, X_test, y_test = load_model_and_data(model_path, X_test_path, y_test_path)
    y_pred = model.predict(X_test)

    # ROC Curve
    plot_roc_curve(model, X_test, y_test, model_name)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

# The main function
def main():
    model_paths = ["./model/LogisticR.pkl", "./model/RandomForest.pkl", "./model/XGBoost.pkl","./model/KNN.pkl"]
    model_names = ["Logistic Regression", "Random Forest", "XGBoost","KNN"]
    X_test_path = "./processed_data/X_test.csv"
    y_test_path = "./processed_data/y_test.csv"

    for model_path, model_name in zip(model_paths, model_names):
        evaluate_model(model_path, X_test_path, y_test_path, model_name)


if __name__ == "__main__":
    main()
