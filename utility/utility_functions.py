import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve,confusion_matrix
import pickle
import numpy as np
from scipy import stats
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 4000)

# A function for removing outliers
def remove_outliers_zscore(df, threshold=3):
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate the z-scores for the columns
    z_scores = np.abs(stats.zscore(df[numeric_columns]))
    
    # Filter out rows where the z-score is greater than the threshold
    filtered_entries = (z_scores < threshold).all(axis=1)

    # Combine the filtered X with the target column and return
    df_cleaned = df[filtered_entries]

    return df_cleaned

def load_data(dataset_path): 
    df = pd.read_csv(dataset_path)

    # Drop columns if they exist in the dataframe
    columns_to_drop = ["PatientID", "DoctorInCharge"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1,inplace=True)
    
    return df

# Load model and test dataset
def load_model_and_data(model_path, X_test_path, y_test_path):

    X_test = load_data(X_test_path)
    y_test = load_data(y_test_path).values.ravel().astype('int')
    
    return load_model(model_path), X_test, y_test

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
    plt.close()

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
    plt.close()


# Save the individual dataset
def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print(f"Datasets saved to {output_dir}")

# Exploratory Data Analysis function
def perform_eda(df,target_column, dataset_name):
    
    # Show the descriptive details about the dataset
    print("".rjust(38, '='))
    print(f"The First 5 Rows of the Dataset")
    print("".rjust(38, '='))
    print(df.head())
    print("".rjust(38, '='))
    print(f"The Last 5 Rows of the Dataset")
    print("".rjust(38, '='))
    print(df.tail())
    print("".rjust(38, '='))
    print(f" The Dimension of the Dataset")
    print("".rjust(38, '='))
    print(df.shape)
    print("".rjust(38, '='))
    print(f" General information about the dataset")
    print("".rjust(38, '='))
    df.info()
    print("".rjust(38, '='))
    print(f" Number of Missing Data By Column")
    print("".rjust(38, '='))
    print(df.isna().sum())
    print("".rjust(38, '='))
    print(f"\n Number of Duplicated Rows")
    print("".rjust(38, '='))
    print(df.duplicated().sum())
    print("".rjust(38, '='))
    print(f"\n Number of Unique Values")
    print("".rjust(38, '='))
    print(df.nunique())
    print("".rjust(38, '='))
    print(f"\n Descriptive statistics:\n")
    print("".rjust(38, '='))
    print(df.describe(include=['float','int']))
    print("".rjust(38, '='))
    print(f"\n Class Imballance Information")
    print(df[target_column].value_counts())
    
    # Obtain plot for the categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for i in categorical_columns:
        barplot(i,df)
    
    if len(categorical_columns)>0:
        # Melt the dataframe to long-form format
        df_melted = df.melt(value_vars=categorical_columns)
        
        # Create a FacetGrid
        g = sns.FacetGrid(df_melted, col="variable", col_wrap=5, sharex=False, sharey=False,aspect=1.5)
        
        # Map the plotting function to the FacetGrid
        g.map_dataframe(countplot_with_order)
        
        # Adjust the layout and save the plot
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{dataset_name}-Count_plot.png')

    # Plot correlation for the numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns)>0:
        # Plot correlation plot
        plt.figure(figsize=(17, 12))
        sns.heatmap(df[numeric_columns].corr(), fmt='.2f',annot=True)
        plt.title(f'Correlation Plot')
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{dataset_name}-Corr_plot.png')
        plt.close()

def countplot_with_order(data, **kwargs):
    col_order = data['value'].value_counts().index
    sns.countplot(x="value", data=data, order=col_order, **kwargs)

# This function imputes missing values for numeric and categorical columns
def handle_missing_values(df):
    
    imputer = SimpleImputer(strategy='median')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    imputer = SimpleImputer(strategy='most_frequent')
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    return df

def handle_class_imbalance(X,y):
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    # Apply label encoding to categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Correct for class imbalance using SMOTE
    oversample = SMOTE()
    X_smoted, y_smoted = oversample.fit_resample(X, y)
    
    # Reverse label encoding
    for col in categorical_columns:
        X_smoted[col] = label_encoders[col].inverse_transform(X_smoted[col])
    
    # return the resampled X and y
    return X_smoted, y_smoted

# Model evaluation function
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):

    #Extract the numeric and categorical columns
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_columns)
        ])
    
    # Define the pipeline
    pipeline = Pipeline([('preprocessor', preprocessor),('clf', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # Obtain performance metrics of the model 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("\n")

    plot_roc_curve(pipeline, X_test, y_test, model_name)

    #Save the pipeline after trainning
    save_model(pipeline,model_name)


def outlier_plot(X):
    
    # Select numeric columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    # Number of columns in the subplot grid
    ncols = len(numeric_columns)
    
    # Number of rows and columns for the grid (r x c)
    r = 6
    c = (ncols + r - 1) // r  # Ensure all plots fit within the grid

    # Create subplots
    f, axes = plt.subplots(r, c, figsize=(20, 18), sharex=True)

    # Flatten axes to iterate over them
    axes = axes.flatten()

    # Create boxplot for each numeric column
    for i, col in enumerate(numeric_columns):
        sns.boxplot(x=X[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
    
    # Remove any unused axes
    for i in range(len(numeric_columns), len(axes)):
        axes[i].axis('off')

    # Create a 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('plots/Outlier_plot.png')
    plt.close()
    
    
def barplot(target, df):

    df = df.drop(['Diagnosis'],axis=1)
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Number of columns in the subplot grid
    ncols = len(numeric_columns)
    
    # Number of rows and columns for the grid (r x c)
    r = 6
    c = (ncols + r - 1) // r  # Ensure all plots fit within the grid

    # Create subplots
    f, axes = plt.subplots(r, c, figsize=(20, 18), sharex=True)

    # Flatten axes to iterate over them
    axes = axes.flatten()

    # Create barplot for each numeric column
    for i, col in enumerate(numeric_columns):
        # Calculate the average of the numeric column grouped by the target
        sns.barplot(x=target, y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} by {target}')
    
    # Remove any unused axes
    for i in range(len(numeric_columns), len(axes)):
        axes[i].axis('off')

    # Create a 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'plots/{target}_Barplot.png')
    plt.close()