
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

    values = [value if value != None else 0 for value in row[2:13]]
    #print values;
    return LabeledPoint(row['Label'], values);



def main():

    #Driver's Program

    #Setting up the standalone mode
    conf = SparkConf().setMaster("local").setAppName("LogisticClassifer")
    sc = SparkContext(conf = conf)

    #Set up dataframe
    sqlContext = SQLContext(sc)

    # Read into Spark Dataframe
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./train.tiny.csv')
    output = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./test.tiny.csv')


    res = df.rdd.map(parsePoint);
    outputRes = output.rdd.map(parsePoint);

    # [class, features ...]

    # Train with 1000 iterations see how classifer converages
    iterations = int(1000)
    model = LogisticRegressionWithSGD.train(res, iterations)

    #Print some weight and intercept, from logistic regression example
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))

    valuesAndPreds = outputRes.map(lambda p:  model.predict(p.features))
    #MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    #print("Mean Squared Error = " + str(MSE))

    #Convert back to dataframe
    #valuesAndPreds.toDF;
    # Output Result
    valuesAndPreds.saveAsTextFile("./result.csv")
    sc.stop()



if __name__ == "__main__": main()
