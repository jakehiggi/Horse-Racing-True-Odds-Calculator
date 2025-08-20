from tqdm import tqdm
import random
import json

from model.horse_model_classif import HorseRaceModelClassif

def evaluate_feature_subsets(
    model_class,
    base_features,
    extra_features,
    data_path,
    feature_engineering_func,
    onehot_columns=None,
    param_grid=None,
    calibrate=True,
    subset_sizes=[3, 5, 8],
    num_trials_per_size=10,
    save_path='feature_subset_results.json'
):
    """
    Iteratively tests different subsets of features to find the best combinations for predictive performance.
    Shows a progress bar via tqdm.
    """
    total_iterations = len(subset_sizes) * num_trials_per_size
    progress_bar = tqdm(total=total_iterations, desc="Evaluating Feature Subsets")

    results = []

    for size in subset_sizes:
        for _ in range(num_trials_per_size):
            subset = random.sample(extra_features, size)
            feature_list = base_features + subset

            model = HorseRaceModelClassif(
                feature_list=feature_list,
                onehot_columns=[col for col in (onehot_columns or []) if col in feature_list],
                feature_engineering_func=feature_engineering_func,
                model_class=model_class,
                model_params=param_grid,
                calibrate=calibrate,
            )

            try:
                model.load_data(data_path)
                model.fit()
                model.predict()
                model.evaluate_against_market()
                model.fit_meta_calibrator()
                model.apply_meta_calibration()

                result = {
                    "Feature Set Size": len(feature_list),
                    "feature_subset": feature_list,
                    "log_loss": model.logloss,
                    "brier_score": model.brier,
                    "market_log_loss": model.market_logloss,
                    "market_brier_score": model.market_brier,
                    "model_accuracy": round(model.model_top_pick_accuracy*100, 3),
                    "market_accuracy": round(model.market_top_pick_accuracy*100, 3),
                    "meta_mae": getattr(model, "meta_mae", None),
                    "meta_top1_accuracy": getattr(model, "meta_accuracy", None)
                }
                results.append(result)
            except Exception as e:
                print(f"Skipping subset {subset} due to error: {e}")
            finally:
                progress_bar.update(1)

    progress_bar.close()

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return results