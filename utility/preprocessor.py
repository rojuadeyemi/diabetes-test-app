def build_preprocessor(X_train):

    ordinal_columns = ['SocioeconomicStatus', 'EducationLevel']
    numeric_columns = X_train.select_dtypes(include=['float64','int64']).columns
    categorical_columns = X_train.select_dtypes(include=['object','category']).columns.difference(ordinal_columns)

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
    from sklearn.impute import SimpleImputer

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('ord', ordinal_transformer, ordinal_columns)
    ])

    preprocessor.set_output(transform="pandas")

    return preprocessor