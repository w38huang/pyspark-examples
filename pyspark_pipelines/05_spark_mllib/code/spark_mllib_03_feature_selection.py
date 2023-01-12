import inspect
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import networkx as nx
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from graphframes import GraphFrame
from graphframes.examples import *
# from graphframes.examples import Graphs

def spark_mllib_01_numeric_features():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_schema = StructType([
        StructField('Pregnancies', DoubleType()),
        StructField('Glucose', DoubleType()),
        StructField('BloodPressure', DoubleType()),
        StructField('SkinThickness', DoubleType()),
        StructField('Insulin', DoubleType()),
        StructField('BMI', DoubleType()),
        StructField('DiabetesPedigreeFunction', DoubleType()),
        StructField('Age', DoubleType()),
        StructField('Outcome', DoubleType())
    ])

    spark = SparkSession.builder.appName('spark_mllib_03_feature_selection').getOrCreate()

    diabetes = spark.read.format("csv").option("header", "true").schema(input_schema).load("../datasets/diabetes.csv")
    diabetes.show()
    diabetes.printSchema()

    ###
    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(inputCols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                                outputCol='features')
    features_outcome = assembler.transform(diabetes)

    ## VectorSlicer
    from pyspark.ml.feature import VectorSlicer
    slicer = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=[1, 2, 3, 4, 5, 7])
    features_subset = slicer.transform(features_outcome)


    ## Univariate Feature Selector https://spark.apache.org/docs/latest/ml-features.html#univariatefeatureselector
    from pyspark.ml.feature import UnivariateFeatureSelector

    selector = UnivariateFeatureSelector(selectionMode="numTopFeatures", featuresCol="features",
                                         outputCol="selectedFeatures", labelCol="Outcome")

    features_outcome.show(3, False)
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(1)
    selected_features = selector.fit(features_outcome).transform(features_outcome)

    selected_features.show()

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_01_numeric_features()

# coding: utf-8



