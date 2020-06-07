import time

import findspark
import pandas as pd
import pyspark as ps
from pymongo import MongoClient
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, StringIndexer, Tokenizer
from pyspark.sql import SQLContext
from pyspark.sql.utils import AnalysisException

from config import password, user
from LG_config import max_iter
import yaml
findspark.init()
try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext()
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")


# Either read csv's or create them and read them
try:
    df1 = (
        sqlContext.read.format("com.databricks.spark.csv")
        .options(header="true", inferschema="true")
        .load("Review_neg.csv")
    )
    df2 = (
        sqlContext.read.format("com.databricks.spark.csv")
        .options(header="true", inferschema="true")
        .load("Review_pos.csv")
    )
except AnalysisException:
    exec(open("./ml/data_handler.py").read())
    df1 = (
        sqlContext.read.format("com.databricks.spark.csv")
        .options(header="true", inferschema="true")
        .load("Review_neg.csv")
    )
    df2 = (
        sqlContext.read.format("com.databricks.spark.csv")
        .options(header="true", inferschema="true")
        .load("Review_pos.csv")
    )


# Concat the two dataframes
df = df1.union(df2)
df.head()
df = df.dropna()


(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed=2000)
start_time = int(round(time.time() * 1000))

tokenizer = Tokenizer(inputCol="review", outputCol="words")
hashtf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol="tf")
idf = IDF(
    inputCol="tf", outputCol="features", minDocFreq=5
)  # minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="label", outputCol="sentiment")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])


pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)

lr = LogisticRegression(maxIter=50)
lrModel = lr.fit(train_df)

predictions = lrModel.transform(val_df)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)
accuracy = predictions.filter(
    predictions.label == predictions.prediction
).count() / float(val_set.count())
timingsss = int(round(time.time() * 1000)) - start_time
print(f"{timingsss} ms")
roc_auc = evaluator.evaluate(predictions)
print(f"{roc_auc:.3f}")


data = {
    "auc" : int(roc_auc * 1000),
    "timing" : timingsss
}

with open(r'spark_stats.yaml', 'w') as file:
    yaml.dump(data, file)