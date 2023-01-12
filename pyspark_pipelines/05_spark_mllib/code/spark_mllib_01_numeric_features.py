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

    spark = SparkSession.builder.appName('spark_mllib_01_numeric_features').getOrCreate()

    diabetes = spark.read.format("csv").option("header", "true").schema(input_schema).load("../datasets/diabetes.csv")
    diabetes.show()
    diabetes.printSchema()

    ### EDA: ???
    # ----- Box plot
    # Plot box plot with keys:outcome, values:age

    # ----- Scatter plot
    # Plot scatter plot with Age and Insulin as Values

    # ------- Bar plot
    # Plot bar graph with Keys: Pregnancies and Values: Glucose

    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(inputCols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                                outputCol='features')

    output = assembler.transform(diabetes)
    diabetes_output = output.select('features', 'Outcome')
    diabetes_output.show()


    ## (01) Standard Scaler: standardize each column as zero mean
    # StandardScaler transforms a dataset of Vector rows, normalizing each feature to have unit standard deviation and/or zero mean
    # https://spark.apache.org/docs/latest/ml-features.html#standardscaler
    from pyspark.ml.feature import StandardScaler

    scaler = StandardScaler(inputCol='features',
                            outputCol='scaledFeatures',
                            withStd=True, withMean=True)

    scaler_model = scaler.fit(diabetes_output)
    diabetes_scaled_data = scaler_model.transform(diabetes_output)
    diabetes_scaled_data.show()


    ## (02) Min Max Scaler
    from pyspark.ml.feature import MinMaxScaler
    scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
    min_max_scaler_model = scaler.fit(diabetes_output)
    diabetes_scaled_data = min_max_scaler_model.transform(diabetes_output)
    diabetes_scaled_data.show()

    ## (03) Normalizer
    from pyspark.ml.feature import Normalizer
    normalizer = Normalizer(inputCol='features', outputCol='normFeatures', p=1.0)
    l1_norm_data = normalizer.transform(diabetes_output)
    l1_norm_data.show()

    ## p2 Normalizer
    normalizer = Normalizer(inputCol='features', outputCol='normFeatures', p=2.0)
    l2_norm_data = normalizer.transform(diabetes_output)

    ## Inf normalizer
    normalizer = Normalizer(inputCol='features', outputCol='normFeatures', p=float("inf"))
    l_inf_norm_data = normalizer.transform(diabetes_output)

    ## (04) Bucketizer
    from pyspark.ml.feature import Bucketizer
    splits = [0, 10, 20, 30, 40, 50, 60, 70, float("inf")]
    bucketizer = Bucketizer(splits=splits, inputCol="Age", outputCol="bucketedAge")
    subset_output = diabetes.select('Age')
    bucketed_data = bucketizer.transform(subset_output)
    bucketed_data.show()

    ## (05) QuantileDiscretizer:
    from pyspark.ml.feature import QuantileDiscretizer
    discretizer = QuantileDiscretizer(numBuckets=5, inputCol="Age", outputCol="discretizedAge")
    discretized_diabetes_data = discretizer.fit(diabetes).transform(diabetes)
    discretized_diabetes_data.select('Age', 'discretizedAge').show()

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_01_numeric_features()

# coding: utf-8



