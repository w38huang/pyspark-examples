import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs

def spark_mllib_05_random_forest_and_gbtregressor():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_schema = StructType([
        StructField("Make", StringType()),
        StructField("Model", StringType()),
        StructField("Vehicle Class", StringType()),
        StructField("Engine Size", DoubleType()),
        StructField("Cylinders", DoubleType()),
        StructField("Transmission", StringType()),
        StructField("Fuel Type", StringType()),
        StructField("Fuel Consumption City", DoubleType()),
        StructField("Fuel Consumption Hwy", DoubleType()),
        StructField(" Fuel Consumption Comb (L/100 km)", DoubleType()),
        StructField("Fuel Consumption Comb (mpg)", DoubleType()),
        StructField("CO2 Emissions(g/km)", DoubleType())
    ])

    spark = SparkSession.builder.appName('spark_mllib_05_RandomForestAndGBTRegressor').getOrCreate()

    co2_raw = (spark.read.format("csv").option("header", "true")
        .schema(input_schema)\
        .load("../datasets/co2.csv")
                    )
    co2_raw.printSchema()
    co2_raw.show(3, False)
    co2_raw.count() ## 7385

    co2 = (co2_raw)
    co2.show()
    co2.printSchema()

    co2.count() ## 2938
    co2 = co2.dropna() ## 1649
    co2.count()

    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
    stages = []
    categoricalColumns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')

        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],
                                outputCols=[categoricalCol + "classVec"])

        stages += [stringIndexer, encoder]

    numericCols = ['Engine Size', 'Cylinders', 'Fuel Consumption City', 'Fuel Consumption Hwy']

    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    stages += [assembler]

    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=stages)

    pipelineModel = pipeline.fit(co2)
    co2_transformed = pipelineModel.transform(co2)

    co2_transformed.select('features', 'CO2 Emissions(g/km)').show()
    co2_train, co2_test = co2_transformed.randomSplit([0.7, 0.3], seed=0)
    print("Training Dataset Count: " + str(co2_train.count()))

    print("Test Dataset Count: " + str(co2_test.count()))


    ## random forrest
    from pyspark.ml.regression import RandomForestRegressor

    rf = RandomForestRegressor(labelCol="CO2 Emissions(g/km)", subsamplingRate=0.8, numTrees=5)
    rfModel = rf.fit(co2_train)
    print("Total numNodes = ", rfModel.totalNumNodes)
    len(rfModel.trees)
    predictions = rfModel.transform(co2_test)
    predictions.select("prediction", "CO2 Emissions(g/km)", "features").show()

    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(labelCol="CO2 Emissions(g/km)", predictionCol="prediction")
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    print("R2 on test data = %g" % r2)

    from pyspark.ml.regression import GBTRegressor
    gbt = GBTRegressor(labelCol="CO2 Emissions(g/km)", maxIter=50)

    gbtModel = gbt.fit(co2_train)
    predictions = gbtModel.transform(co2_test)
    predictions.select("prediction", "CO2 Emissions(g/km)", "features").show()
    evaluator = RegressionEvaluator(labelCol="CO2 Emissions(g/km)", predictionCol="prediction")
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    print("R2 on test data = %g" % r2)

    import pyspark.sql.functions as F

    predictions_with_residuals = predictions.withColumn("residual",
                                                        (F.col("CO2 Emissions(g/km)") - F.col("prediction")))

    (predictions_with_residuals.agg({'residual': 'mean'}).show())

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_05_random_forest_and_gbtregressor()

# coding: utf-8



