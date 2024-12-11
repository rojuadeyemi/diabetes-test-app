from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from utility.utility_functions import load_data,train_and_evaluate_model

# Train all the models using the dataset in the specified path
def train_models(X_train_path, X_test_path, y_train_path, y_test_path):

    
    X_train, X_test = load_data(X_train_path), load_data(X_test_path)
    
    y_train, y_test  = load_data(y_train_path), load_data(y_test_path)
    
    y_train = y_train.values.ravel().astype('int')
    y_test = y_test.values.ravel().astype('int')
    models =[LogisticRegression(max_iter=1000,warm_start=True,n_jobs=-1),
             RandomForestClassifier(n_estimators=100,warm_start=True,n_jobs=-1),
             XGBClassifier(eval_metric='logloss',n_jobs=-1),KNeighborsClassifier(n_jobs=-1)]

    model_names = ["LogisticR","RandomForest","XGBoost","KNN"]
    for model,model_name in zip(models,model_names):
        train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name)


if __name__ == "__main__":
    X_train_path = './processed_data/X_train.csv'
    X_test_path = './processed_data/X_test.csv'
    y_train_path = './processed_data/y_train.csv'
    y_test_path = './processed_data/y_test.csv'
    train_models(X_train_path, X_test_path, y_train_path, y_test_path)
