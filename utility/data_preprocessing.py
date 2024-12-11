from utility.utility_functions import (split_train_test_data, load_data, save_datasets,
                                    perform_eda, handle_missing_values, remove_outliers_zscore,
                                    handle_class_imbalance)
    
def main(file_path, target_column, output_dir):

    #Load the dataset
    df = load_data(file_path)

    # Perform EDA
    perform_eda(df, target_column, "Raw Dataset")

    # Handle missing data, if there's any
    df = handle_missing_values(df)
    
    # Remove outliers
    df = remove_outliers_zscore(df)
    
    #Split the dataset
    X_train, X_test, y_train, y_test = split_train_test_data(df, target_column)

    #handle class imbalance
    X_train, y_train = handle_class_imbalance(X_train,y_train)

    # Save the datasets
    save_datasets(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    file_path = "./raw/diabetes_data.csv" 
    target_column = "Diagnosis"
    output_dir = "./processed_data" 
    main(file_path, target_column, output_dir)
