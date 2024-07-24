import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pickle
import os
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 4000)

def load_data(dataset_path):
    df = pd.read_csv(dataset_path).drop(["PatientID","DoctorInCharge"],axis=1)
    return df

#Split the Dataset into train and test datasets
def load_train_test_data(df,target_variable,test_size = 0.2):
    y = df[target_variable]
    X=df.drop([target_variable],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=24)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test


# Save the model
def save_model(model, model_name):
    os.makedirs('./model', exist_ok=True)
    # store the model as model_name.pkl
    with open(f"./model/{model_name}.pkl","wb") as file:
        pickle.dump(model,file)


# Load the model
def load_model(model_path):
    with open(model_path,"rb") as file:
        model = pickle.load(file)
    return model


# Function to plot ROC
def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2%})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="best")
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_roc_curve.png')

    return roc_auc

# Confusion Matrix function
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred,normalize = 'true')
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.0%', xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')


# Save the individual dataset
def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print(f"Datasets saved to {output_dir}")

