from sklearn.model_selection import RandomizedSearchCV
from utility.model_registry import models, cv, scoring,param_grids
from utility.model_pipeline import build_pipeline
from utility.preprocessor import build_preprocessor
from utility.utility_functions import save_model
from sklearn import set_config
set_config(transform_output="pandas")

# Train all the models and cross-validate, then save the best model
def model_tuning(X_train, y_train):

    y_train = y_train.values.ravel().astype(int)

    results_log = []

    best_score = -1
    best_name = None

    preprocessor = build_preprocessor(X_train)

    for name, model in models.items():

        pipeline = build_pipeline(model,preprocessor)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=100,
            cv=cv,
            scoring=scoring,
            refit="precision",
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the model
        search.fit(X_train, y_train)

        cv_res = search.cv_results_

        row = {
            "model": name,
            "roc_auc": cv_res["mean_test_roc_auc"][search.best_index_],
            "recall": cv_res["mean_test_recall"][search.best_index_],
            "precision": cv_res["mean_test_precision"][search.best_index_],
            "f1": cv_res["mean_test_f1"][search.best_index_],
            "accuracy": cv_res["mean_test_balanced_accuracy"][search.best_index_]
        }

        results_log.append(row)

        print(f"{name}: {search.best_score_}")

        # Save the model
        save_model(search.best_estimator_, f"cv_{name}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            
    import pandas as pd
    results_df = pd.DataFrame(results_log).sort_values("precision", ascending=False)

    results_df.to_csv("./report/model_comparison.csv", index=False)

    print(f"\nBest Model: {best_name} | Score: {best_score}")

