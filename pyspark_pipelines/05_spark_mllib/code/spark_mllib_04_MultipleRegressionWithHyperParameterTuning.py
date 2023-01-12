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
        StructField("Country", StringType()),
        StructField("Year", DoubleType()),
        StructField("Status", StringType()),
        StructField("Life expectancy ", DoubleType()),
        StructField("Adult Mortality", DoubleType()),
        StructField("infant deaths", DoubleType()),
        StructField("Alcohol", DoubleType()),
        StructField("percentage expenditure", DoubleType()),
        StructField("Hepatitis B", DoubleType()),
        StructField("Measles ", DoubleType()),
        StructField(" BMI ", DoubleType()),
        StructField("under-five deaths ", DoubleType()),
        StructField("Polio", DoubleType()),
        StructField("Total expenditure", DoubleType()),
        StructField("Diphtheria ", DoubleType()),
        StructField(" HIV/AIDS", DoubleType()),
        StructField("GDP", DoubleType()),
        StructField("Population", DoubleType()),
        StructField(" thinness  1-19 years", DoubleType()),
        StructField(" thinness 5-9 years", DoubleType()),
        StructField("Income composition of resources", DoubleType()),
        StructField("Schooling", DoubleType())
    ])

    spark = SparkSession.builder.appName('spark_mllib_04_MultipleRegressionWithHyperParameterTuning').getOrCreate()

    life_exp_raw = spark.read.format("csv").option("header", "true").schema(input_schema).load("../datasets/life_expectancy.csv")
    life_exp_raw.printSchema()
    life_exp_raw.show(3, False)


    life_exp = (life_exp_raw


                )
    life_exp.show()
    life_exp.printSchema()

    life_exp.count() ## 2938
    life_exp = life_exp.dropna() ## 1649
    life_exp.count()

    independent_variables = ['Adult Mortality',
                             'Schooling',
                             'Total expenditure',
                             'Diphtheria ',
                             'GDP',
                             'Population']

    dependent_variable = ['Life expectancy ']

    ## calculate correlation:
    life_exp_corr = life_exp.select(independent_variables + dependent_variable)

    for i in life_exp_corr.columns:
        print("Correlation to life expectancy for", i, "is: ", life_exp_corr.stat.corr('Life expectancy ', i))

    from pyspark.ml.feature import StringIndexer
    categoricalCols = ['Status']
    stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols])

    stringIndexerModel = stringIndexer.fit(life_exp)

    life_exp_statusindex = stringIndexerModel.transform(life_exp)

    life_exp_statusindex.filter(life_exp_statusindex['Country'].isin(['Afghanistan', 'Germany'])) \
        .select('Country', 'Status', 'StatusIndex') \
        .show()

    feature_columns = ['Year', 'Adult Mortality', 'infant deaths',
                       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
                       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                       ' thinness 5-9 years', 'Income composition of resources', 'Schooling', 'StatusIndex']

    label_column = 'Life expectancy '

    from pyspark.ml.feature import VectorAssembler

    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    life_exp_features_label = vector_assembler.transform(life_exp_statusindex).select(['features', label_column])

    life_exp_features_label.show()

    train_df, test_df = life_exp_features_label.randomSplit([0.75, 0.25], seed=123)

    # train_df.count(), test_df.count() # (1227, 422)

    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    linear_regression = LinearRegression(featuresCol='features', labelCol=label_column)
    linear_regression_model = linear_regression.fit(train_df)

    print('Model Coefficients: \n' + str(linear_regression_model.coefficients))
    training_summary = linear_regression_model.summary
    print('RMSE: %f' % training_summary.rootMeanSquaredError)
    print('R-SQUARED: %f' % training_summary.r2)
    print('TRAINING DATASET RESIDUALS: ')

    training_summary.residuals.show()

    test_predictions = linear_regression_model.transform(test_df)
    print('TEST DATASET PREDICTIONS AGAINST ACTUAL LABEL: ')
    test_predictions.select('features', 'prediction', 'life expectancy ').show()

    test_summary = linear_regression_model.evaluate(test_df)
    print('RMSE on Test Data = %g' % test_summary.rootMeanSquaredError)
    print('R-SQUARED on Test Data = %g' % test_summary.r2)


    ## Hyper parameter Tuning
    from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

    paramGrid = ParamGridBuilder() \
        .addGrid(linear_regression.regParam, [0.1, 0.05, 0.01]) \
        .addGrid(linear_regression.fitIntercept, [False, True]) \
        .addGrid(linear_regression.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    evaluator = RegressionEvaluator(labelCol=label_column)
    tvs = TrainValidationSplit(estimator=linear_regression,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               trainRatio=0.8)
    model = tvs.fit(train_df)
    tuned_prediction = model.transform(test_df)

    tuned_prediction.select('features', 'Life expectancy ', 'prediction').show()

    r2_score = evaluator.setMetricName('r2').evaluate(tuned_prediction)
    print('R-SQUARED on Test Data = %g' % r2_score)

    print('Best regParam: ' + str(model.bestModel._java_obj.getRegParam()) + "\n" +
          'Best ElasticNetParam:' + str(model.bestModel._java_obj.getElasticNetParam()))

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_01_numeric_features()

# coding: utf-8



