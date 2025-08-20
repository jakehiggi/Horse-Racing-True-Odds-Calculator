import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from scipy.special import softmax
import json
import random
from pathlib import Path

class HorseRaceModelClassif:
    def __init__(self, feature_list, model_class=XGBClassifier, model_params=None, 
                 use_cv=True, calibrate=True, feature_engineering_func=None, onehot_columns=None, 
                 random_state=19):
        self.feature_list = feature_list
        self.model_class = model_class
        self.model_params = model_params #or {}
        self.use_cv = use_cv
        self.calibrate = calibrate
        self.feature_engineering_func = feature_engineering_func
        self.onehot_columns = onehot_columns or []
        self.random_state = random_state
        self.pipeline = None
        self.model = None
        self.best_params = None
        self.logloss = None
        self.brier = None
        self.market_logloss = None
        self.market_brier = None
        self.val_df = None
        self.final_features_used = []
        self.raw_data_path = None
        self.horse_factor_map = {}
        self.feature_importances = {}


    def _default_feature_engineering(self, df):
        df['SpeedRank'] = df.groupby('Race_ID')['Speed_PreviousRun'].rank(pct=True)
        df['OddsRatio_LTO'] = df['MarketOdds_PreviousRun'] / df.groupby('Race_ID')['MarketOdds_PreviousRun'].transform('mean')
        df['TrainerRel'] = df['TrainerRating'] / df.groupby('Race_ID')['TrainerRating'].transform('mean')
        df['JockeyRel'] = df['JockeyRating'] / df.groupby('Race_ID')['JockeyRating'].transform('mean')
        df['PrizePerRunner'] = df['Prize'] / df['Runners']
        df['OddsTrend'] = df['MarketOdds_PreviousRun'] - df['MarketOdds_2ndPreviousRun']
        df['SpeedDelta'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        return df

    def load_data(self, path):
        self.raw_data_path = path
        df = pd.read_csv(path)
        df['target'] = (df['Position'] == 1).astype(int)

        for col in df.select_dtypes('object').columns:
            if col == 'Horse':
                df[col], self.horse_factor_map = pd.factorize(df[col])
            else:
                df[col] = pd.factorize(df[col])[0]

        df.drop(columns=['betfairSP', 'timeSecs', 'pdsBeaten', 'NMFP', 'Position'], errors='ignore', inplace=True)

        # Apply feature engineering
        df = self.feature_engineering_func(df) if self.feature_engineering_func else self._default_feature_engineering(df)

        # Validate feature list (after feature engineering)
        unsafe = {'target', 'Race_ID'}
        used_features = [f for f in self.feature_list if f in df.columns]
        unsafe_used = [f for f in used_features if f in unsafe]
        if unsafe_used:
            raise ValueError(f"Feature list contains ID/leakage-prone columns: {unsafe_used}")

        race_ids = df['Race_ID'].unique()
        train_races, val_races = train_test_split(race_ids, test_size=0.2, random_state=self.random_state)
        train_df = df[df['Race_ID'].isin(train_races)].copy()
        val_df = df[df['Race_ID'].isin(val_races)].copy()

        self.X_train = train_df[used_features]
        self.X_val = val_df[used_features]
        self.y_train = train_df['target']
        self.y_val = val_df['target']
        self.final_features_used = used_features
        self.val_df_full = val_df
        return df

    def fit(self):
        numeric_cols = [col for col in self.feature_list if col not in self.onehot_columns]
        transformers = []

        if numeric_cols:
            transformers.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols))

        if self.onehot_columns:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.onehot_columns))

        preprocessor = ColumnTransformer(transformers=transformers)

        model_params = self.model_params if self.model_params is not None else {}
        base_model = self.model_class(**{k.replace("classifier__", ""): v for k, v in model_params.items()})
        model_pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', base_model)
        ])
        
        if self.use_cv:
            if self.model_params is None:
                param_grid = {
                    'classifier__n_estimators': [100, 300, 500],           # Number of boosting rounds
                    'classifier__max_depth': [3, 5, 7, 9],                 # Max tree depth
                    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],   # Shrinkage
                    'classifier__subsample': [0.6, 0.8, 1.0],              # Row sampling
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0],       # Feature sampling per tree
                    'classifier__gamma': [0, 0.1, 0.2, 0.4],               # Minimum loss reduction
                    'classifier__reg_alpha': [0, 0.01, 0.1, 1],            # L1 regularization
                    'classifier__reg_lambda': [0.1, 1, 5],                 # L2 regularization
                }
            else:
                param_grid = self.model_params

            search = RandomizedSearchCV(model_pipeline, param_grid, n_iter=30, scoring='neg_log_loss', cv=2, random_state=self.random_state)
            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            model_pipeline.fit(self.X_train, self.y_train)
            best_model = model_pipeline
            self.best_params = self.model_params

        # Force a fresh calibration/model each time
        if self.calibrate:
            self.model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3) # < ----- can try isotonic instead of sigmoid
            self.model.fit(self.X_train, self.y_train)
        else:
            self.model = best_model 

    def predict(self, test_df=None):
        if test_df is not None:
            for col in test_df.select_dtypes('object').columns:
                if col == 'Horse' \
                and getattr(self, 'horse_factor_map', None) is not None \
                and len(self.horse_factor_map) > 0:
                    # use the exact same category ordering you learned during train/load_data
                    test_df[col] = pd.Categorical(
                        test_df[col],
                        categories=self.horse_factor_map
                    ).codes
                else:
                    # factorize everything else from scratch
                    test_df[col], _ = pd.factorize(test_df[col])

            test_df = self.feature_engineering_func(test_df) if self.feature_engineering_func else self._default_feature_engineering(test_df)
            test_X = test_df[self.final_features_used]
            test_df = test_df.copy()
            test_df['raw_prob'] = self.model.predict_proba(test_X)[:, 1]
            test_df['Predicted_Probability'] = (
                test_df['raw_prob'] / test_df.groupby('Race_ID')['raw_prob'].transform('sum')
            )
            
            out = test_df[['Race_ID', 'Horse', 'Predicted_Probability']].copy()
            out['Horse'] = out['Horse'].map(lambda c: self.horse_factor_map[c])
            return out
        else:
            self.val_df = self.val_df_full.copy()
            self.val_df['raw_prob'] = self.model.predict_proba(self.X_val)[:, 1]
            self.val_df['Predicted_Probability'] = (
                self.val_df['raw_prob'] / self.val_df.groupby('Race_ID')['raw_prob'].transform('sum')
            )
            self.logloss = log_loss(self.val_df['target'], self.val_df['Predicted_Probability'])
            self.brier = brier_score_loss(self.val_df['target'], self.val_df['Predicted_Probability'])
            
            out = self.val_df[['Race_ID', 'Horse', 'Predicted_Probability']].copy()
            out['Horse'] = out['Horse'].map(lambda c: self.horse_factor_map[c])
            return self.val_df[['Race_ID', 'Horse', 'Predicted_Probability']]

   
    def evaluate_against_market(self, odds_column="betfairSP"):
        if self.val_df is None:
            raise ValueError("Run .predict() before evaluating against market.")

        raw_df = pd.read_csv(self.raw_data_path)
        if 'Horse' in raw_df.columns:
            #raw_df['Horse'] = pd.Series(pd.Categorical(raw_df['Horse'], categories=self.horse_factor_map)).codes
            raw_df['Horse'] = pd.Categorical(raw_df['Horse'], categories=self.horse_factor_map).codes
        raw_df = raw_df[raw_df['Race_ID'].isin(self.val_df['Race_ID'].unique())][['Race_ID', 'Horse', odds_column]]

        self.val_df = pd.merge(self.val_df, raw_df, on=['Race_ID', 'Horse', ], how='left')
        self.val_df['market_prob'] = 1 / self.val_df[odds_column]
        self.val_df['market_prob'] = self.val_df.groupby('Race_ID')['market_prob'].transform(lambda x: x / x.sum())

        self.market_logloss = log_loss(self.val_df['target'], self.val_df['market_prob'])
        self.market_brier = brier_score_loss(self.val_df['target'], self.val_df['market_prob'])

        print("Model vs. Market Comparison:")
        print(f"Model Log Loss:  {self.logloss:.4f}")
        print(f"Market Log Loss: {self.market_logloss:.4f}")
        print(f"Model Brier:     {self.brier:.4f}")
        print(f"Market Brier:    {self.market_brier:.4f}")

        model_top = self.val_df.loc[self.val_df.groupby('Race_ID')['Predicted_Probability'].idxmax()]
        market_top = self.val_df.loc[self.val_df.groupby('Race_ID')['market_prob'].idxmax()]

        # Check if the top predicted horse actually won
        model_top['model_correct'] = model_top['target'] == 1
        market_top['market_correct'] = market_top['target'] == 1

        # Calculate accuracy
        model_accuracy = model_top['model_correct'].mean()
        market_accuracy = market_top['market_correct'].mean()

        self.model_top_pick_accuracy = model_accuracy
        self.market_top_pick_accuracy = market_accuracy

        print(f"\nModel top pick accuracy: {model_accuracy*100:.2f}%")
        print(f"Market top pick accuracy: {market_accuracy*100:.2f}%")

        random_race_id = random.choice(self.val_df['Race_ID'].unique())
        print(self.val_df[self.val_df['Race_ID'] == random_race_id][['Race_ID', 'Horse', 'Predicted_Probability', 'market_prob', 'target']])
        print(self.val_df[self.val_df['Race_ID'] == random_race_id]['Predicted_Probability'].sum())

        MAE = mean_absolute_error(self.val_df['market_prob'], self.val_df['Predicted_Probability'])
        print("Mean Absolute Error vs Market:", MAE)


    def evaluate_feature_importance(self, scoring='neg_log_loss', n_repeats=5, top_n=None):
        if self.model is None:
            raise ValueError("Model must be fitted before evaluating feature importance.")

        print("Evaluating feature importance with permutation importance...")
        result = permutation_importance(
            self.model, self.X_val, self.y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.random_state
        )

        importances = {
            feature: round(score, 6)
            for feature, score in zip(self.final_features_used, result.importances_mean)
        }

        # Sort by importance descending
        self.feature_importances = dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))

        if top_n:
            print(f"Top {top_n} features by importance:")
            for k, v in list(self.feature_importances.items())[:top_n]:
                print(f"{k}: {v}")
        return self.feature_importances


    def fit_meta_calibrator(self, extra_features=None):
        """
        Trains a more advanced meta-calibration model using model output, rank, and optionally market data.
        Only run this AFTER .predict() and .evaluate_against_market().
        """
        if self.val_df is None:
            raise ValueError("You must run .predict() first.")
        if 'market_prob' not in self.val_df.columns:
            raise ValueError("Run .evaluate_against_market() before advanced meta calibration.")

        df = self.val_df.copy()
        df['rank'] = df.groupby('Race_ID')['raw_prob'].rank(ascending=False, pct=True)
        df['log_raw_prob'] = np.log(df['raw_prob'] + 1e-8)
        df['market_rank'] = df.groupby('Race_ID')['market_prob'].rank(ascending=False, pct=True)

        meta_features = ['raw_prob', 'rank', 'log_raw_prob', 'market_prob', 'market_rank']
        if extra_features:
            meta_features += extra_features

        X_meta = df[meta_features]
        y_meta = df['target']

        from xgboost import XGBRegressor
        self.advanced_meta_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05)
        self.advanced_meta_model.fit(X_meta, y_meta)

        print("Advanced meta-calibration model fitted.")

    def apply_meta_calibration(self, test_df=None):
        """
        Apply advanced meta-calibration to validation or test data.
        Requires market_prob to be available.
        """
        if not hasattr(self, 'advanced_meta_model'):
            raise ValueError("Run .fit_advanced_meta_calibrator() first.")

        df = self.val_df.copy() if test_df is None else test_df.copy()

        if 'market_prob' not in df.columns:
            raise ValueError("Market probabilities required. Run .evaluate_against_market() first.")

        df['rank'] = df.groupby('Race_ID')['raw_prob'].rank(ascending=False, pct=True)
        df['log_raw_prob'] = np.log(df['raw_prob'] + 1e-8)
        df['market_rank'] = df.groupby('Race_ID')['market_prob'].rank(ascending=False, pct=True)

        meta_features = ['raw_prob', 'rank', 'log_raw_prob', 'market_prob', 'market_rank']
        X_meta = df[meta_features]

        df['meta_prob'] = self.advanced_meta_model.predict(X_meta)
        df['Predicted_Probability'] = df['meta_prob'] / df.groupby('Race_ID')['meta_prob'].transform('sum')

        if test_df is not None:
            return df[['Race_ID', 'Horse', 'Predicted_Probability']]
        else:
            self.val_df['meta_prob'] = df['meta_prob']
            self.val_df['Predicted_Probability'] = df['Predicted_Probability']
            self.logloss = log_loss(self.val_df['target'], self.val_df['Predicted_Probability'])
            self.brier = brier_score_loss(self.val_df['target'], self.val_df['Predicted_Probability'])
            self.meta_mae = mean_absolute_error(self.val_df['market_prob'], self.val_df['Predicted_Probability']) if 'market_prob' in self.val_df else None
            self.meta_accuracy = accuracy_score(
                self.val_df['target'],
                self.val_df.groupby('Race_ID')['Predicted_Probability'].transform(lambda x: x == x.max()).astype(int)
            )
            
            print("Advanced meta calibration applied.")
            print(f"Log Loss:         {self.logloss:.4f}")
            print(f"Brier Score:      {self.brier:.4f}")
            if self.meta_mae is not None:
                 print(f"MAE vs Market:    {self.meta_mae:.4f}")
            print(f"Top-1 Accuracy:   {self.meta_accuracy:.4%}")
            #return self.val_df[['Race_ID', 'Horse', 'Predicted_Probability']]

    def compare_calibration_sources(self, race_id=None):
        """
        Visualizes raw model probabilities, meta-calibrated probabilities, and market probabilities
        for a single race. If no race_id is provided, picks one at random.
        """
        import matplotlib.pyplot as plt
        df = self.val_df.copy()
        if race_id is None:
            race_id = np.random.choice(df['Race_ID'].unique())

        df_race = df[df['Race_ID'] == race_id].copy()
        df_race.sort_values('Predicted_Probability', ascending=False, inplace=True)

        plt.figure(figsize=(10, 5))
        bar_width = 0.2
        x = np.arange(len(df_race))

        plt.bar(x - bar_width, df_race['raw_prob'], width=bar_width, label='Raw Prob')
        if 'meta_prob' in df_race.columns:
            plt.bar(x, df_race['meta_prob'], width=bar_width, label='Meta-Calibrated')
        if 'market_prob' in df_race.columns:
            plt.bar(x + bar_width, df_race['market_prob'], width=bar_width, label='Market')

        plt.xticks(x, df_race["Horse"].astype(str), rotation=45)
        plt.ylabel("Probability")
        plt.title(f"Probability Comparison for Race {race_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def save_results(self, path="horse_model_results.json"):
        results = {
            "features_used": self.final_features_used,
            "feature_importances": self.feature_importances,

            "log_loss": self.logloss,
            "brier_score": self.brier,
            "market_log_loss": self.market_logloss,
            "market_brier_score": self.market_brier,

            "market_top": f"{self.market_top_pick_accuracy*100:.2f}%",
            "model_top_pick_accuracy": f"{self.model_top_pick_accuracy*100:.2f}%",

            "best_params": self.best_params
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        return results



