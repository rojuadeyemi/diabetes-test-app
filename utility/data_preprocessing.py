import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility.utility_functions import load_train_test_data,load_data,save_datasets
from sklearn.impute import SimpleImputer
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Exploratory Data Analysis function
def perform_eda(df,dataset_name):
    
    print(f"{dataset_name} First 5 rows:\n", df.head())
    print(f"\n General information about the dataset:\n", df.info())
    print(f"\n Missing data:\n", df.isnull().sum())
    print(f"\n Descriptime statistics:\n", df.describe())
    
    # Obtain plot for the categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
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

    # Plot histogram for the the last 4 continuous columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns)>0:
        # Plot correlation plot
        plt.figure(figsize=(17, 12))
        sns.heatmap(df[numeric_columns].corr(), fmt='.2f',annot=True)
        plt.title(f'Correlation Plot')
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{dataset_name}-Corr_plot.png')

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

def handle_class_imbalance(df,target_column):
    
    y = df[target_column]
    X = df.drop([target_column], axis=1)
    
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
    
    # Combine the resampled X and y into a single DataFrame
    return pd.concat([X_smoted, y_smoted], axis=1)
    
def main(file_path, target_column, output_dir):

    #Load the dataset
    df = load_data(file_path)

    # Perform EDA
    perform_eda(df,"Raw Dataset")

    # Handle missing data, if there's any
    df = handle_missing_values(df)

    #handle class imbalance
    df = handle_class_imbalance(df,target_column)

    #Split the dataset
    X_train, X_test, y_train, y_test = load_train_test_data(df, target_column)
    save_datasets(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    file_path = "./raw/diabetes_data.csv" 
    target_column = "Diagnosis"
    output_dir = "./processed_data" 
    main(file_path, target_column, output_dir)
