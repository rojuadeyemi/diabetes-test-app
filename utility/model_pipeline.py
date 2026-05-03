from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

def build_pipeline(model, preprocessor):

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ("smote", SMOTE(random_state=24)),
        ("feature_selection", SelectFromModel(model, threshold="median")),
        ('classifier', model)
    ])
    return pipeline
    