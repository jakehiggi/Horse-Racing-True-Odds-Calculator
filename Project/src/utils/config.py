from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
TEST_RESULTS_DIR = BASE_DIR / 'model_testing_results'

TRAIN_CSV_PATH = DATA_DIR / 'trainData.csv'
TEST_CSV_PATH = DATA_DIR / 'testData.csv'
