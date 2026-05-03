from sklearn.base import BaseEstimator, TransformerMixin

# Feature Engineer Wrapper Class
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = feature_engineering(X)
        self.feature_names_out_ = X_new.columns
        return X_new

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

# Extract Polynomial Features
def add_polynomial_features(df, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    poly = PolynomialFeatures(degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_columns])
    poly_feature_names = poly.get_feature_names_out(numeric_columns)

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Remove original columns from polynomial output
    poly_df = poly_df.drop(columns=numeric_columns, errors='ignore')

    df = df.reset_index(drop=True)
    poly_df = poly_df.reset_index(drop=True)

    return pd.concat([df, poly_df], axis=1)

# This function obtains all new features for the pipeline
def feature_engineering(df):
    
    df = df.copy()

    # 1. Add 2nd order polinomial    
    df = add_polynomial_features(df)

    return df
