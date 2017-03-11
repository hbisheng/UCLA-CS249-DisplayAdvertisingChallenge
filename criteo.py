import sys

from random import randint


from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext


from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel,LogisticRegressionWithSGD



def parsePoint(row):

    """
    Parse each row of text into an MLlib LabeledPoint object.
    """
    values = []
    for value in row[1:40]:
        if value == None or value == '':
            values.append(0)
        else:
            values.append(int(str(value), 16))
    return LabeledPoint(row[0], values);


def main():

    #Driver's Program
    #Setting up the standalone mode
    conf = SparkConf().setMaster("local").setAppName("LogisticClassifer")
    sc = SparkContext(conf = conf)

    #Set up dataframe
    sqlContext = SQLContext(sc)

    # Read into Spark Dataframe
    train_data = sqlContext.read.format('com.databricks.spark.csv').options(inferschema='true', delimiter="\t").load('./train.txt')

    train_parsed = train_data.rdd.map(parsePoint);
    
    # [class, features ...]

    # Train with 1000 iterations see how classifer converages
    iterations = int(1000)
    model = LogisticRegressionWithSGD.train(train_parsed, iterations)

    #Print some weight and intercept, from logistic regression example
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))

    training_correct = train_parsed.map(lambda p: 1 if model.predict(p.features) == p.label else 0)
    training_acc = training_correct.sum() * 1.0 / train_parsed.count()
    
    print "training acc: " + str(training_acc)

    #MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    #print("Mean Squared Error = " + str(MSE))

    #Convert back to dataframe
    #valuesAndPreds.toDF;
    # Output Result
    train_parsed.map(lambda p: model.predict(p.features)).saveAsTextFile("./train_labels.csv")
    sc.stop()

if __name__ == "__main__": main()