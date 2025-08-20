# ------ IMPORTS ------ #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autofeat import AutoFeatClassifier
from sklearn.feature_selection import mutual_info_classif

from utils.config import TRAIN_CSV_PATH
from utils.feature_engineer import make_feat

# ------- DATA CONFIG ------ #
df = pd.read_csv(TRAIN_CSV_PATH)
df_full = make_feat(df)

print(df_full.head())