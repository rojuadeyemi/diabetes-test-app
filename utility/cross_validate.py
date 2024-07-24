from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from utility.utility_functions import load_train_test_data,load_data,save_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def grid_search_val(model,X_train,y_train,param_grid,cv,scoring):

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Best parameters and best score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC Score: {grid_search.best_score_}")

    # Update the model with best parameters
    best_model = grid_search.best_estimator_

    return best_model
    
# Train all the models and cross-validate, then save the best model
def main(dataset_path,y,test_size,cv,scoring):
    
    df = load_data(dataset_path)
    X_train, X_test, y_train, y_test = load_train_test_data(df, y, test_size)
    y_train = y_train.values.ravel().astype('int')
    y_test = y_test.values.ravel().astype('int')

    # Extract the numeric and categorical columns
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Define preprocessor for numeric and categorical columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_columns)
        ])

    model_names = ["LogisticR", "RandomForest", "XGBoost","KNN"]
    models = [
        LogisticRegression(warm_start=True, n_jobs=-1),
        RandomForestClassifier(warm_start=True, n_jobs=-1),
        XGBClassifier(eval_metric='logloss', n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1)
    ]

    pipe_models = {}
    results = {}

    for model, model_name in zip(models, model_names):
        pipeline = Pipeline([('preprocessor', preprocessor), ('clf', model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        results[model_name] = scores
        pipe_models[model_name] = pipeline

    # Determine the best model based on mean score
    best_model_name = max(results, key=lambda k: results[k].mean())
    best_model = pipe_models[best_model_name]

    # Define parameter grids for grid search
    param_grids = {
        "LogisticR": {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['liblinear', 'lbfgs']
        },
        "RandomForest": {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7]
        },
        "XGBoost": {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        },
        "KNN": {
            'clf__n_neighbors': [3, 5, 7],
            'clf__weights': ['uniform', 'distance'],
            'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'clf__p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
        }
        }
    # Perform grid search for the best model
    final_best_model = grid_search_val(best_model, X_train, y_train, param_grid=param_grids[best_model_name], cv=cv, scoring=scoring)

    # Save the best model
    save_model(final_best_model, "cv_" + best_model_name)


if __name__ == "__main__":
    dataset_path = "./raw/diabetes_data.csv"
    y='Diagnosis'
    test_size=0.2
    cv=5
    scoring='roc_auc'
    main(dataset_path, y, test_size,cv,scoring)
