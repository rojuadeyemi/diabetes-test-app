import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# Extract Polynomial Features
def add_polynomial_features(df, degree=2):
    poly = PolynomialFeatures(degree, include_bias=False)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    poly_features = poly.fit_transform(df[numeric_columns])
    poly_feature_names = poly.get_feature_names_out(numeric_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    df = df.reset_index(drop=True)
    poly_df = poly_df.reset_index(drop=True)

    df = pd.concat([df, poly_df], axis=1)
    return df


# Main function
def main(train_file_path, test_file_path, train_output_path,test_output_path, degree=2):
    
    # Load the train dataset and store in the specified path
    df_train = pd.read_csv(train_file_path)
    df_train = add_polynomial_features(df_train, degree)
    df_train.to_csv(train_output_path, index=False)

    # Load the test dataset and store in the specified path
    df_test = pd.read_csv(test_file_path)
    df_test = add_polynomial_features(df_test, degree)
    df_test.to_csv(test_output_path, index=False)


if __name__ == "__main__":
    train_file_path = "./processed_data/X_train.csv" 
    test_file_path = "./processed_data/X_test.csv" 
    train_output_path = "./processed_data/X_train_engineered.csv"
    test_output_path = "./processed_data/X_test_engineered.csv" 
    degree = 2  # Polynomial degree
    main(train_file_path, test_file_path, train_output_path, test_output_path, degree)
