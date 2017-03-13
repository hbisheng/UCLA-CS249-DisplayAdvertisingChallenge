# pyspark --packages com.databricks:spark-csv_2.10:1.2.0
# spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 criteo.py
import datetime
import sys
import shutil

from random import randint
from math import exp, log, sqrt

from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import *
from pyspark.mllib.linalg import SparseVector

BUCKET_SIZE = 2 ** 20

def parsePoint(row):
    """
    Parse each row of text into an MLlib LabeledPoint object.
    """
    dict = {}
    for column in row[1:40]:
        if column == None or column == '':
            pass
        else:
            hash_val = (int(str(column), 16)) % BUCKET_SIZE # dumb hashing
            dict[hash_val] = 1

    return LabeledPoint(row[0], SparseVector(BUCKET_SIZE, dict))


# probability of p(y = 1 | x; w)
def get_probability(wTx):
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

def trunc_float(f):
    return float("%.4f" % f)

def main():
    #Driver's Program
    #Setting up the standalone mode
    conf = SparkConf().setMaster("local").setAppName("LogisticClassifer")
    sc = SparkContext(conf = conf)

    #Set up dataframe
    sqlContext = SQLContext(sc)

    # Read into Spark Dataframe
    train_data = sqlContext.read.format('com.databricks.spark.csv').options(inferschema='true', delimiter="\t").load('./train.txt')
    test_data  = sqlContext.read.format('com.databricks.spark.csv').options(inferschema='true', delimiter="\t").load('./test_with_id.txt')
    
    train_parsed = train_data.rdd.map(parsePoint);
    test_parsed  = test_data.rdd.map(parsePoint)
    model = LogisticRegressionWithSGD.train(train_parsed, iterations=100)
    
    #Print some weight and intercept, from logistic regression example
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))

    #training_correct = train_parsed.map(lambda p: 1 if model.predict(p.features) == p.label else 0)
    #training_acc = training_correct.sum() * 1.0 / train_parsed.count()
    #print "training acc: " + str(training_acc)
    
    shutil.rmtree("./train_labels_and_scores", ignore_errors=True)
    shutil.rmtree("./test_score", ignore_errors=True)

    model.clearThreshold()
    train_parsed.map(
        lambda point: ( trunc_float(model.predict(point.features)), point.label, trunc_float(logloss(model.predict(point.features), point.label) ) )  ) \
            .saveAsTextFile("./train_labels_and_scores")

    print "Training Logloss: " + str(
        train_parsed.map(
            lambda point: logloss(model.predict(point.features), point.label)
            ).sum() / train_parsed.count())

    test_parsed \
        .sortBy(lambda point: point.label) \
        .map(lambda point: str(int(point.label)) + "," + str(  get_probability(model.predict(point.features)))) \
        .saveAsTextFile("./test_score")
    
    #MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    #print("Mean Squared Error = " + str(MSE))
    #Convert back to dataframe
    #valuesAndPreds.toDF;
    #Output Result
    
    sc.stop()

if __name__ == "__main__": main()