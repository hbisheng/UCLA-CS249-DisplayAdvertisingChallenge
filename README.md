1. Install the Spark 2.1.0, and Numpy (in order to support the Mllib)

2. Put the code and data in the right path

3. Specify the path of the input data in the code, and choose the hash method to be used in the parsePoint function

4. Run the command "spark-submit criteo_dist_regression.py" to train a model using distributed logistic regression or "spark-submit criteo_mllib_regression.py" to train a model using Spark Mllib.
