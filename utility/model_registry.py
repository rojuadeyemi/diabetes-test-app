from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint, uniform, loguniform


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["precision", "roc_auc", "f1","balanced_accuracy","recall"]

models = {
    "logistic": LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000),
    "rf": RandomForestClassifier(max_depth=4, random_state=42),
    "xgb": XGBClassifier(max_depth=4, random_state=42)
}

param_grids = {
    "logistic": {
        "classifier__C": loguniform(1e-4, 1),
    },
    "rf": {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 20),
        'classifier__min_samples_leaf': randint(1, 10),
        'classifier__max_features': ['sqrt', 'log2']
        
    },
    "xgb": {
        'classifier__n_estimators': randint(100, 500),
        'classifier__max_depth': randint(3, 20),
        'classifier__learning_rate': loguniform(0.01, 0.2),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.7, 0.3),
        'classifier__min_child_weight': randint(1, 5),
        'classifier__gamma': uniform(0, 0.3),
        'classifier__reg_alpha': loguniform(1e-4, 10),
        'classifier__reg_lambda': loguniform(1e-4, 10)
    }
}