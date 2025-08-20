
import pandas as pd

from xgboost import XGBClassifier

from model.horse_model_classif import HorseRaceModelClassif
from utils.feature_engineer import make_feat, engineered_feature_list
from utils.config import BASE_DIR, DATA_DIR, TEST_RESULTS_DIR, TEST_CSV_PATH, TRAIN_CSV_PATH
from utils.evaluate_subsets import evaluate_feature_subsets
from utils.best_features import best_features_list


# ------------------------ FEATURE SELECTION ------------------------ #
extra_features = engineered_feature_list()
base_features = [
    'Runners', 
    'NMFPLTO',
    'DistanceForm_Rank',
    'HJ_Pair_Count',
    'Avg_NMFPLTO_Cond',
    'DistanceSpecialization',
]
onehot_cols = [
    'Going',
    'Distance_Category',
    'GoingFactor',
 ]

best_features_list = best_features_list()

# ------------------------ CONFIGURATION ------------------------  #
"""
Configuration for a single run of the model. 
"""
# TEST_RESULT_NAME = \
#                    "LAST RUN.json"
# json_path = TEST_RESULTS_DIR / TEST_RESULT_NAME

# model = HorseRaceModelClassif(
#     feature_list= best_features_list,
#     model_class=XGBClassifier,
#     model_params= {
#         'classifier__n_estimators': [50, 100, 200],
#         'classifier__max_depth': [3, 5, 7],
#         'classifier__learning_rate': [0.01, 0.05, 0.1],
#         'classifier__subsample': [0.6, 0.8, 1.0],
#         'classifier__colsample_bytree': [0.6, 0.8, 1.0]
#     },
#     use_cv=True,
#     onehot_columns=onehot_cols,
#     random_state=25,
#     feature_engineering_func=make_feat
# )

# model.load_data(csv_path)
# model.fit()
# predictions = model.predict() 
# model.evaluate_against_market() 
# model.evaluate_feature_importance(top_n=len(seed_test_columns))
# model.fit_meta_calibrator()
# model.apply_meta_calibration()
# model.compare_calibration_sources()
# model.save_results(json_path)


# ------------------------ EVALUATE FEATURE SUBSETS ------------------------  #
"""
Evaluate different subsets of features to find the best combinations for predictive performance.
This will test various combinations of features and save the results to a JSON file.
"""
# evaluate_feature_subsets(
#     model_class=XGBClassifier,
#     base_features=base_features,
#     extra_features=extra_features,
#     data_path=TRAIN_CSV_PATH,
#     param_grid= {
#         'classifier__n_estimators': [50, 100, 200],
#         'classifier__max_depth': [3, 5, 7],
#         'classifier__learning_rate': [0.01, 0.05, 0.1],
#         'classifier__subsample': [0.6, 0.8, 1.0],
#         'classifier__colsample_bytree': [0.6, 0.8, 1.0]
#     },
#     feature_engineering_func=make_feat,
#     onehot_columns=onehot_cols,
#     subset_sizes=[8, 9, 10, 11, 12],
#     num_trials_per_size=50,
#     save_path='subset_eval_results_NO2.json'
# )



# ------------------------ RUN MODEL WITH BEST FEATURES ------------------------  #
"""
Run the model using the best features identified from previous evaluations.
"""
model = HorseRaceModelClassif(
    feature_list=best_features_list,
    onehot_columns=['Going'],
    model_class=XGBClassifier,
    model_params= {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0]
    },
    calibrate=True,
    use_cv=True,
    random_state=19,
    feature_engineering_func=make_feat
)


model.load_data(TRAIN_CSV_PATH)
model.fit()
model.predict()
model.evaluate_against_market()
model.fit_meta_calibrator()
model.apply_meta_calibration()

test_df = pd.read_csv(TEST_CSV_PATH)

predictions_df = model.predict(test_df=test_df)
predictions_df.to_csv(
    BASE_DIR / "test_predictions_new.csv",
    index=True
)

