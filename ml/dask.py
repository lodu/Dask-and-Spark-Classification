import time

import joblib
import pandas as pd
from dask.distributed import Client
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from helper import DescribeClassifier

from config import user, password

from LG_config import max_iter, max_features


mongo_client = MongoClient(
    f"mongodb://{user}:{password}@localhost:27017/?authSource=admin&readPreference=primary"
)
db = mongo_client.reviews

result = db.cleaned.find({})
source = list(result)
df = pd.DataFrame(source)
df.head()

client = None


if __name__ == "__main__":
    client = Client()
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.2
    )

    vec = TfidfVectorizer(stop_words={"english"}, max_features=max_features)
    X_train_vec = vec.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=max_iter, fit_intercept=True, intercept_scaling=5,
    )

    param_grid = {
        "C": [0.001, 10.0],
        # liblinear doesnt make sense (linear)
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
    }

    with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
        grid_search = GridSearchCV(model, param_grid, verbose=2)
        grid_search.fit(X_train_vec, y_train)
        print(grid_search.best_params_, grid_search.best_score_)
