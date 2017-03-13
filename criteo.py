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

    TRAIN_INPUT_FILE = './train_100k.txt'
    TEST_INPUT_FILE  = './test_with_id.txt'

    TRAIN_OUTPUT_FILE = "./train_labels_and_scores"
    TEST_OUTPUT_FILE  = "./test_score"
    MODEL_OUTPUT_FILE = "./model_linear_regression"
    LOGLOSS_OUTPUT_FILE = "./res_logloss"

    # Be sure to backup!
    shutil.rmtree(TRAIN_OUTPUT_FILE, ignore_errors=True)
    shutil.rmtree(TEST_OUTPUT_FILE, ignore_errors=True)
    shutil.rmtree(MODEL_OUTPUT_FILE, ignore_errors=True)
    shutil.rmtree(LOGLOSS_OUTPUT_FILE, ignore_errors=True)

    print datetime.datetime.now()
    conf = SparkConf().setMaster("local").setAppName("LogisticClassifer")
    sc = SparkContext(conf = conf)

    train_data = sc.textFile(TRAIN_INPUT_FILE).map( lambda p: p.split('\t') )
    test_data  = sc.textFile(TEST_INPUT_FILE).map( lambda p: p.split('\t'))

    train_parsed = train_data.map(parsePoint);
    test_parsed  = test_data.map(parsePoint)

    model = LogisticRegressionWithSGD.train(train_parsed, iterations=100)
    model.save(sc, MODEL_OUTPUT_FILE)

    print "Model trained"
    print datetime.datetime.now()

    #training_correct = train_parsed.map(lambda p: 1 if model.predict(p.features) == p.label else 0)
    #training_acc = training_correct.sum() * 1.0 / train_parsed.count()
    #print "training acc: " + str(training_acc)
    
    model.clearThreshold()

    train_parsed.map(
        lambda point: ( trunc_float(model.predict(point.features)), point.label, trunc_float(logloss(model.predict(point.features), point.label) ) )  ) \
            .saveAsTextFile(TRAIN_OUTPUT_FILE)

    avg_logloss = train_parsed.map(
            lambda point: logloss(model.predict(point.features), point.label)
            ).sum() / train_parsed.count()

    print "training logloss: " + str(avg_logloss)
    sc.parallelize([avg_logloss]).repartition(1).saveAsTextFile(LOGLOSS_OUTPUT_FILE)

    test_parsed \
        .sortBy(lambda point: point.label) \
        .map(lambda point: str(int(point.label)) + "," + str(  get_probability(model.predict(point.features)))) \
        .repartition(1) \
        .saveAsTextFile(TEST_OUTPUT_FILE)
    
    print datetime.datetime.now()
    sc.stop()

if __name__ == "__main__": main()