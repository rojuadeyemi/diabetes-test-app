from utility.utility_functions import (load_data, save_datasets,
                                    perform_eda, handle_missing_values,
                                    handle_class_imbalance,outlier_plot,remove_outliers_zscore)
from sklearn.model_selection import train_test_split

def main(data_path, target_column, output_dir,test_size):

    #Load the dataset
    df = load_data(data_path)

    # Perform EDA
    perform_eda(df, target_column, "Raw Dataset")

    # Handle missing data, if there's any
    df = handle_missing_values(df)

    # remove outliers
    df = remove_outliers_zscore(df)
    
    y = df[target_column]
    X=df.drop([target_column],axis=1)

    # Check the outlier plot
    outlier_plot(X)
    
    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)

    #handle class imbalance
    X_train, y_train = handle_class_imbalance(X_train,y_train)

    # Save the datasets
    save_datasets(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    data_path = "./raw/diabetes_data.csv" 
    target_column = "Diagnosis"
    output_dir = "./processed_data"
    test_size = 0.2
    main(data_path, target_column, output_dir,test_size)
