import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs

def spark_mllib_07_DecisionTreeClassification():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_schema = StructType([
        StructField("age", IntegerType()),
        StructField("job", StringType()),
        StructField("marital", StringType()),
        StructField("education", StringType()),
        StructField("default", StringType()),
        StructField("balance", IntegerType()),
        StructField("housing", StringType()),
        StructField("loan", StringType()),
        StructField("contact", StringType()),
        StructField("day", IntegerType()),
        StructField("month", StringType()),
        StructField("duration", IntegerType()),
        StructField("campaign", IntegerType()),
        StructField("pdays", IntegerType()),
        StructField("previous", IntegerType()),
        StructField("poutcome", StringType()),
        StructField("deposit", StringType())
    ])

    spark = SparkSession.builder.appName('spark_mllib_05_RandomForestAndGBTRegressor').getOrCreate()


    bank_raw = (spark.read.format("csv").option("header", "true")
        .schema(input_schema)\
        .load("../datasets/bank.csv")
                    )
    bank_raw.printSchema()
    bank_raw.show(3, False)
    bank_raw.count() ## 11162

    bank = (bank_raw.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                   'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit'))
    bank.show()
    bank.printSchema()

    bank.count() ## 2938
    bank = bank.dropna() ## 1649
    bank.count()

    numeric_features = [t[0] for t in bank.dtypes if t[1] == 'int']
    (bank.select(numeric_features).describe().show())

    ## Feature Selection
    bank = bank.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                       'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')

    cols = bank.columns
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

    stages = []
    categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')

        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],
                                outputCols=[categoricalCol + "classVec"])

        stages += [stringIndexer, encoder]

    label_stringIndexer = StringIndexer(inputCol='deposit', outputCol='label')

    stages += [label_stringIndexer]

    numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="originalFeatures")

    stages += [assembler]
    from pyspark.ml.feature import UnivariateFeatureSelector

    selector = UnivariateFeatureSelector(selectionMode="numTopFeatures", featuresCol="originalFeatures",
                                         outputCol="features", labelCol="label")

    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(20)
    stages += [selector]

    from pyspark.ml import Pipeline

    pipeline = Pipeline(stages=stages)

    pipelineModel = pipeline.fit(bank)

    bank_transformed = pipelineModel.transform(bank)

    bank_transformed.select('features', 'label').show()
    bank_train, bank_test = bank_transformed.randomSplit([0.7, 0.3], seed=2018)
    print("Training Dataset Count: " + str(bank_train.count()))

    print("Test Dataset Count: " + str(bank_test.count()))

    from pyspark.ml.classification import DecisionTreeClassifier
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=3)
    dtModel = dt.fit(bank_train)
    print("numNodes = ", dtModel.numNodes)
    print("depth = ", dtModel.depth)

    dtPreds = dtModel.transform(bank_test)

    dtPreds.select('age', 'job', 'rawPrediction', 'prediction', 'probability', 'label').show()
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    dtEval = BinaryClassificationEvaluator()
    dtEval.evaluate(dtPreds)
    print(dt.explainParams())

    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

    paramGrid = (ParamGridBuilder()
                 .addGrid(dt.maxDepth, [1, 3, 6, 10])
                 .addGrid(dt.maxBins, [20, 40, 80, 100])
                 .build())
    cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=dtEval, numFolds=5)
    cvModel = cv.fit(bank_train)
    print("numNodes = ", cvModel.bestModel.numNodes)
    print("depth = ", cvModel.bestModel.depth)
    cvPreds = cvModel.transform(bank_test)

    cvPreds.select('label', 'prediction').show()
    print(dtEval.evaluate(cvPreds))

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_07_DecisionTreeClassification()

# coding: utf-8



