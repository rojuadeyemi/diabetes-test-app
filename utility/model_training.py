from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from utility.utility_functions import load_train_test_data,save_model,load_data,plot_roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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


# Train all the models using the dataset in the specified path
def train_models(dataset_path, y, test_size):

    df = load_data(dataset_path)
    X_train, X_test, y_train, y_test = load_train_test_data(df, y, test_size)
    
    y_train = y_train.values.ravel().astype('int')
    y_test = y_test.values.ravel().astype('int')
    models =[LogisticRegression(max_iter=1000,warm_start=True,n_jobs=-1),
             RandomForestClassifier(n_estimators=100,warm_start=True,n_jobs=-1),
             XGBClassifier(eval_metric='logloss',n_jobs=-1),KNeighborsClassifier(n_jobs=-1)]

    model_names = ["LogisticR","RandomForest","XGBoost","KNN"]
    for model,model_name in zip(models,model_names):
        train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name)


if __name__ == "__main__":
    dataset_path = "./raw/diabetes_data.csv" 
    y='Diagnosis'
    test_size=0.2
    train_models(dataset_path, y, test_size)
