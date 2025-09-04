from src.data_prep.preprocess import load_dataset


def train_model():
    X_train, X_test, y_train, y_test = load_dataset()

    print(X_train.head())

