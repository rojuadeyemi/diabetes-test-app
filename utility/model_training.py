from utility.model_pipeline import build_pipeline
from utility.preprocessor import build_preprocessor
from utility.model_registry import models, cv, scoring
from utility.utility_functions import save_model
from sklearn.model_selection import cross_validate
from sklearn import set_config

set_config(transform_output="pandas")

# Train and evaluate each of the models using the dataset
def train_and_evaluate_model(X, y):

    y_train = y.values.ravel().astype('int')

    preprocessor = build_preprocessor(X)

    for model_name, model in models.items():

        pipeline = build_pipeline(model, preprocessor)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )    

        print(f"Model: {model_name}")
        for eval_type, title in {"test_":"Validation","train_":"Training"}.items():
            print(f"{title} Performance")
            print(f"ROC AUC: {scores[f'{eval_type}roc_auc'].mean():.2f}")
            print(f"Precision: {scores[f'{eval_type}precision'].mean():.2f}")
            print(f"Recall: {scores[f'{eval_type}recall'].mean():.2f}")
            print(f"f1: {scores[f'{eval_type}f1'].mean():.2f}")
            print(f"Accuracy: {scores[f'{eval_type}balanced_accuracy'].mean():.2f}")
            print("\n")

        # Fit the model
        pipeline.fit(X, y_train)

        #Save the pipeline after trainning
        save_model(pipeline,model_name)