import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs

def spark_mllib_06_streaming_data_classification():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_schema = StructType([
        StructField("ID", StringType()),
        StructField("Default", DoubleType()),
        StructField("Loan_type", StringType()),
        StructField("Gender", StringType()),
        StructField("Age", DoubleType()),
        StructField("Degree", StringType()),
        StructField("Income", DoubleType()),
        StructField("Credit_score", DoubleType()),
        StructField("Loan_length", DoubleType()),
        StructField("Signers", DoubleType()),
        StructField("Citizenship", StringType())
    ])

    spark = SparkSession.builder.appName('spark_mllib_05_RandomForestAndGBTRegressor').getOrCreate()


    loan_raw = (spark.read.format("csv").option("header", "true")
        .schema(input_schema)\
        .load("../datasets/loan_data/loan_data.csv")
                    )
    loan_raw.printSchema()
    loan_raw.show(3, False)
    loan_raw.count() ## 7385

    loan = (loan_raw)
    loan.show()
    loan.printSchema()

    loan.count() ## 2938
    loan = loan.dropna() ## 1649
    loan.count()

    loan = loan.drop("ID") ## 4569

    train_df, test_df = loan.randomSplit([0.8, 0.2], seed=0)
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer
    stages = []
    categoricalColumns = ['Loan_type', 'Gender', 'Citizenship', 'Degree']

    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')

        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],
                                outputCols=[categoricalCol + "classVec"])

        stages += [stringIndexer, encoder]

    numericCols = ['Age', 'Income', 'Credit_score', 'Loan_length', 'Signers']

    assemblerInputs = [c + 'classVec' for c in categoricalColumns] + numericCols

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol='rawFeatures')

    indexer = VectorIndexer(inputCol='rawFeatures', outputCol='features', maxCategories=8)

    stages += [assembler, indexer]

    from pyspark.ml.classification import LogisticRegression

    lr = LogisticRegression(featuresCol='features', labelCol='Default', regParam=1.0)

    stages += [lr]

    from pyspark.ml import Pipeline

    pipeline = Pipeline(stages=stages)

    p_model = pipeline.fit(train_df)

    test_pred = p_model.transform(test_df)

    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

    bcEvaluator = BinaryClassificationEvaluator(labelCol='Default', metricName="areaUnderROC")

    print(f"Area under ROC curve: {bcEvaluator.evaluate(test_pred)}")

    mcEvaluator = MulticlassClassificationEvaluator(labelCol='Default', metricName="accuracy")

    print(f"Accuracy: {mcEvaluator.evaluate(test_pred)}")

    loan_pred_data = (spark.readStream.format("csv") \
        .option("header", True) \
        .schema(input_schema) \
        .option("ignoreLeadingWhiteSpace", True) \
        .load("../datasets/loan_data/prediction_data/")
        )

    ### How to show schema data ???
    # loan_pred_data.writeStream.format("console").outputMode("complete").start()
    # loan_pred_data.show(3, False)

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_06_streaming_data_classification()

# coding: utf-8



