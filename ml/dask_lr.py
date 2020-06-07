import time

import joblib
import pandas as pd
import yaml
from dask.distributed import Client
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from config import password, user
from LG_config import max_features, max_iter

mongo_client = MongoClient(
    f"mongodb://{user}:{password}@localhost:27017/?authSource=admin&readPreference=primary"
)
db = mongo_client.reviews

result = db.short.find({})
source = list(result)
df = pd.DataFrame(source)
df.head()
client = None

def main():
    client = Client()
    X_train, X_test, y_train, y_test = train_test_split(
        df["Review"].astype(str), df["Sentiment"], test_size=0.2
    )

    with joblib.parallel_backend("dask", n_jobs=6):
        start_time = int(round(time.time() * 1000))

        vec = TfidfVectorizer(stop_words={"english"}, max_features=max_features)
        X_train_vec = vec.fit_transform(X_train)

        model = LogisticRegression(
            max_iter=50,
            fit_intercept=True,
            intercept_scaling=5,
            C=1,
            solver="newton-cg",
        )

        model.fit(X_train_vec, y_train)
        X_test_vec = vec.transform(X_test)
        y_pred = model.predict(X_test_vec)
        timingss = int(round(time.time() * 1000)) - start_time
        print(f"{timingss} ms")

        auc = roc_auc_score(y_test, y_pred)
        print(f"{auc:.3f}")

        data =  {
        "auc" : int(auc * 1000),
        "timing" : timingss
        }
        with open(r'dask_stats.yaml', 'w') as file:
            yaml.dump(data, file)
# Otherwise
if __name__ == "__main__":
        main()
        
        
