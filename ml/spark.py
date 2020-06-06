import findspark

import pyspark as ps
from config import password, user
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, StringIndexer, Tokenizer
from pyspark.sql import SQLContext
from LG_config import max_iter

findspark.init()
try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext()
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")


df = (
    sqlContext.read.format("com.databricks.spark.csv")
    .options(header="true", inferschema="true")
    .load("xd.csv")
)
df = df.dropna()


(train_set, val_set, test_set) = df.randomSplit([0.98, 0.01, 0.01], seed=2000)

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
train_df.show(5)

lr = LogisticRegression(maxIter=max_iter)
lrModel = lr.fit(train_df)

predictions = lrModel.transform(val_df)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)
accuracy = predictions.filter(
    predictions.label == predictions.prediction
).count() / float(val_set.count())
roc_auc = evaluator.evaluate(predictions)
print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))
