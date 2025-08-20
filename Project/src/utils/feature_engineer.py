import pandas as pd
import numpy as np

def make_feat(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # ----------------------------------------------------------------------
    # 1. Date / Time Features
    # ----------------------------------------------------------------------
    X["Race_Time_Parsed"] = pd.to_datetime(
        X["Race_Time"], dayfirst=True, errors="coerce"
    )

    X["Hour"] = X["Race_Time_Parsed"].dt.hour
    X["Month"] = X["Race_Time_Parsed"].dt.month

    # ‑‑‑ helper mappers -----------------------------------------------------
    def _map_time_of_day(hour: float | int ) -> str:
        if pd.isna(hour):
            return "Unknown"
        if hour < 12:
            return "Morning"
        if hour < 17:
            return "Afternoon"
        if hour < 21:
            return "Evening"
        return "Night"

    def _map_season(month: float | int ) -> str:
        if pd.isna(month):
            return "Unknown"
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    # categorical → One‑Hot encode later
    X["TimeOfDay"] = X["Hour"].apply(_map_time_of_day)
    X["Season"] = X["Month"].apply(_map_season)

    # ----------------------------------------------------------------------
    # 2. Distance & Course Features
    # ----------------------------------------------------------------------
    X["distance_f"] = X["distanceYards"] / 220  # yards → furlongs

    # Average of last two speed figures
    X["avg_speed"] = X[["Speed_PreviousRun", "Speed_2ndPreviousRun"]].mean(axis=1)

    # Estimated time (simple heuristic)
    X["estimated_time"] = X["distance_f"] / X["avg_speed"].replace(0, np.nan)

    # Distance bucket (categorical → One‑Hot later)
    X["Distance_Category"] = pd.cut(
        X["distance_f"],
        bins=[0, 6, 10, 20],
        labels=["Sprint", "Mid", "Stayer"],
    )

    # Distance‑specific form (horse × distance bucket)
    distance_perf = (
        X.groupby(["Horse", "Distance_Category"], observed=False)["NMFPLTO"]
        .mean()
        .reset_index()
        .rename(columns={"NMFPLTO": "Avg_NMFPLTO_inType"})
    )
    X = X.merge(distance_perf, on=["Horse", "Distance_Category"], how="left")

    # Course affinity (horse × course)
    course_perf = (
        X.groupby(["Horse", "Course"])["NMFPLTO"].mean().reset_index().rename(
            columns={"NMFPLTO": "Avg_NMFPLTO_Course"}
        )
    )
    X = X.merge(course_perf, on=["Horse", "Course"], how="left")

    # ----------------------------------------------------------------------
    # 3. Speed‑related Features
    # ----------------------------------------------------------------------
    X["relative_Speed_LTO"] = (
        X["Speed_PreviousRun"]
        - X.groupby("Race_ID")["Speed_PreviousRun"].transform("mean")
    )

    speed_grp = X.groupby("Race_ID")["Speed_PreviousRun"]
    X["Speed_Zscore"] = (
        X["Speed_PreviousRun"] - speed_grp.transform("mean")
    ) / speed_grp.transform("std")
    X["Speed_vs_Top"] = X["Speed_PreviousRun"] - speed_grp.transform("max")
    X["Speed_vs_Median"] = X["Speed_PreviousRun"] - speed_grp.transform("median")

    # Momentum between last two runs
    X["Speed_Momentum"] = (
        X["Speed_PreviousRun"] - X["Speed_2ndPreviousRun"]
    )
    X["SpeedMomentum_Percentile"] = X.groupby("Race_ID")["Speed_Momentum"].rank(
        pct=True
    )

    # Going adjustment
    going_factor = X.groupby("Going")["avg_speed"].mean().to_dict()
    X["GoingFactor"] = X["Going"].map(going_factor)
    X["Adjusted_Speed_Going"] = X["avg_speed"] / X["GoingFactor"]

    # ----------------------------------------------------------------------
    # 4. Jockey & Trainer Features
    # ----------------------------------------------------------------------
    X["jockey_percentile"] = X.groupby("Race_ID")["JockeyRating"].rank(pct=True)
    X["trainer_percentile"] = X.groupby("Race_ID")["TrainerRating"].rank(pct=True)

    X["is_top_jockey"] = (
        X.groupby("Race_ID")["JockeyRating"].transform("max")
        == X["JockeyRating"]
    ).astype(int)

    X["TrainerRating_Relative"] = X["TrainerRating"] / X.groupby("Race_ID")[
        "TrainerRating"
    ].transform("mean")
    X["JockeyRating_Rank"] = X.groupby("Race_ID")["JockeyRating"].rank(pct=True)

    # Trainer × jockey interaction
    X["trainer_jockey_strength"] = X["TrainerRating"] * X["JockeyRating"]

    # Horse–jockey familiarity
    pair_counts = (
        X.groupby(["Horse", "Jockey"]).size().reset_index(name="HJ_Pair_Count")
    )
    X = X.merge(pair_counts, on=["Horse", "Jockey"], how="left")

    # ----------------------------------------------------------------------
    # 5. Form & Momentum Features
    # ----------------------------------------------------------------------
    # Previous‑form trend (2‑race rolling mean of NMFPLTO)
    X_sorted = X.sort_values(["Horse", "Race_Time_Parsed"])
    X_sorted["NMFPLTO_Trend"] = (
        X_sorted.groupby("Horse")["NMFPLTO"]
        .shift(1)
        .rolling(window=2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    X["NMFPLTO_Trend"] = X_sorted["NMFPLTO_Trend"].values
    X["NMFPLTO_Trend"] = X["NMFPLTO_Trend"].fillna(X["NMFPLTO"])

    X["NMFP_Rank"] = X.groupby("Race_ID")["NMFPLTO"].rank(pct=True)

    # Market‑odds momentum
    X["delta_market_odds"] = (
        X["MarketOdds_PreviousRun"] - X["MarketOdds_2ndPreviousRun"]
    )
    X["pct_change_market_odds"] = (
        X["delta_market_odds"] / X["MarketOdds_2ndPreviousRun"].replace(0, np.nan)
    )
    X["Odds_Momentum"] = (
        X["MarketOdds_2ndPreviousRun"] - X["MarketOdds_PreviousRun"]
    )

    # ----------------------------------------------------------------------
    # 6. Distance‑form & Specialisation (race‑relative)
    # ----------------------------------------------------------------------
    X["DistanceForm_Rank"] = X.groupby("Race_ID")["Avg_NMFPLTO_inType"].rank(
        pct=True
    )

    avg_nmfplto_overall = X.groupby("Horse")["NMFPLTO"].transform("mean")
    X["DistanceSpecialization"] = (
        X["Avg_NMFPLTO_inType"] - avg_nmfplto_overall
    )

    # Race‑level z‑scores
    rank_grp = X.groupby("Race_ID")["DistanceForm_Rank"]
    X["DistanceForm_Zscore"] = (
        X["DistanceForm_Rank"] - rank_grp.transform("mean")
    ) / rank_grp.transform("std")

    spec_grp = X.groupby("Race_ID")["DistanceSpecialization"]
    X["DistanceSpecialization_Z"] = (
        X["DistanceSpecialization"] - spec_grp.transform("mean")
    ) / spec_grp.transform("std")

    # Distance‑form vs best in race
    X["DistanceForm_vs_Top"] = X["Avg_NMFPLTO_inType"] - X.groupby("Race_ID")[
        "Avg_NMFPLTO_inType"
    ].transform("max")

    # ----------------------------------------------------------------------
    # 7. Miscellaneous Features
    # ----------------------------------------------------------------------
    X["relative_Age"] = X["Age"] - X.groupby("Race_ID")["Age"].transform("mean")
    X["prize_per_runner"] = X["Prize"] / X["Runners"]

    # Ordinal bin; treat as numeric or one‑hot
    X["days_since_race_bin"] = pd.cut(
        X["daysSinceLastRun"], bins=[0, 14, 30, 90, 365], labels=False
    )

    # ----------------------------------------------------------------------
    # 8. Conditional‑form Features (Horse × Distance × Going)
    # ----------------------------------------------------------------------
    cond_perf = (
        X.groupby(["Horse", "Distance_Category", "Going"], observed=False)[
            "NMFPLTO"
        ]
        .mean()
        .reset_index()
        .rename(columns={"NMFPLTO": "Avg_NMFPLTO_Cond"})
    )
    X = X.merge(
        cond_perf, on=["Horse", "Distance_Category", "Going"], how="left"
    )

    # ----------------------------------------------------------------------
    # 9. Data‑quality Checks
    # ----------------------------------------------------------------------
    if "Avg_NMFPLTO_inType" not in X.columns:
        raise ValueError(
            "Avg_NMFPLTO_inType not created during merge — check 'Horse' or 'Distance_Category'"
        )

    # ----------------------------------------------------------------------
    # 10. Categorical columns to ONE‑HOT Encode
    # ----------------------------------------------------------------------
    # ‑ 'Going'
    # ‑ 'Distance_Category'
    # ‑ 'Season'
    # ‑ 'TimeOfDay'
    # ‑ 'Course'  (optional: high cardinality → consider target/leave‑one‑out enc.)
    # ‑ 'Horse', 'Jockey', 'Trainer' (very high cardinality; embeddings or leave as IDs)
    # ----------------------------------------------------------------------

    return X

def engineered_feature_list():
    lst = ['jockey_percentile',
        'relative_Age',
        'trainer_jockey_strength',
        'prize_per_runner',
        'Adjusted_Speed_Going',
        'days_since_race_bin',
        'Avg_NMFPLTO_inType', 
        #'DistanceForm_Rank',
        #'DistanceSpecialization',
        'Speed_Zscore',
        'Speed_vs_Top',
        'TrainerRating_Relative',
        #'Avg_NMFPLTO_Cond',
        'Speed_vs_Median',
        'Avg_NMFPLTO_Course',
        #'HJ_Pair_Count',
        'DistanceForm_vs_Top',
        #'DistanceSpecialization_Z',
        #'Distance_Specialization2',
        #'Distance_Category', # <----- OneHot Encoding required
        'is_top_jockey',
        'avg_speed',
        'delta_market_odds',
        'estimated_time',
        'pct_change_market_odds', 
        'NMFPLTO_Trend',
        'Speed_Momentum',
        'SpeedMomentum_Percentile',
        'JockeyRating_Rank',
        'trainer_percentile',
        'GoingFactor',
        'relative_Speed_LTO',
        'DistanceForm_Zscore',
        'Odds_Momentum',
        'distance_f',
        'NMFP_Rank',
        'TrainerRating', 'Going', #'Course',
        'daysSinceLastRun', 'SireRating', 'DamsireRating',
        'Speed_PreviousRun', 'Speed_2ndPreviousRun', 'Age', 'MarketOdds_2ndPreviousRun',
        'MarketOdds_PreviousRun'
        ]
    return lst