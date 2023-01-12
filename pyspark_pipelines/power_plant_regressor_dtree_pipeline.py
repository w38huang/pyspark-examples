import inspect

from pyspark.ml import Pipeline
from pyspark.ml.regression import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col

def run():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_path = 'data/common/power_plant/csv/*.csv'

    # step 1: create a sparksession
    spark = create_sparksession()

    # step 2: read input dataframe
    rawDf = read_data(spark, input_path)

    rawDf.printSchema()
    rawDf.show(10, False)

    # step 3: preprocessing
    df = preprocess(rawDf)

    # step 4: train test split
    (train, test) = df.randomSplit([0.7, 0.3], seed=100)

    # step5: build ml pipeline
    # rawLabelColumn = "class"
    labelColumn = "PE"
    featuresColumn = "features"
    predictionColumn = "Predicted_PE"
    predictedLabel = "predictedLabel"

    vectorAssembler = create_feature_pipeline(df, labelColumn, featuresColumn)

    # step 6: choose a model
    model = DecisionTreeRegressor().setFeaturesCol(featuresColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)

    # step 7: create a model pipeline
    stages = [vectorAssembler, model]
    pipeline = Pipeline().setStages(stages)

    # step 8: setup hyper parameter tuning parameters
    params = ParamGridBuilder().addGrid(model.impurity, ["variance"]).addGrid(model.maxDepth, [1, 2, 3, 4, 5]).build()

    # step 9: setup evaluation metrics: f1
    evaluator = RegressionEvaluator().setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    rmseEvaluator = evaluator.setMetricName("rmse")

    # step 10: train validation split
    tvs = (TrainValidationSplit()
           .setTrainRatio(0.75)
           .setSeed(123456)
           .setEstimatorParamMaps(params)
           .setEstimator(pipeline)
           .setEvaluator(rmseEvaluator))

    tvsModel = tvs.fit(train)

    # step 11: get bestPipelineModel
    bestPipelineModel = tvsModel.bestModel

    # step 12: get bestModel
    bestModel = bestPipelineModel.stages[0]

    # step 13: predict on test dataset
    testPredictedDf = bestPipelineModel.transform(test)
    testMetrics = modelEvaluator(evaluator, testPredictedDf)

    # step 14: predict on train dataset
    trainPredictedDf = bestPipelineModel.transform(train)
    trainMetrics = modelEvaluator(evaluator, trainPredictedDf)

    print(trainMetrics)
    print(testMetrics)

    # step 15: calculate all Model Parameters
    paramsAndMetrics = calcAllModelParamsAndMetrics(tvsModel)

    print("hello")


def create_feature_pipeline(df, labelColumn, featuresColumn):
    # # inputCols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    inputCols = df.columns
    inputCols.remove(labelColumn)

    vectorAssembler = VectorAssembler().setInputCols(inputCols).setOutputCol(featuresColumn)

    return (vectorAssembler)

def preprocess(rawDf):
    df = (rawDf.select(
        col("AT").cast("Double"),
        col("V").cast("Double"),
        col("AP").cast("Double"),
        col("RH").cast("Double"),
        col("PE").cast("Double")
    ))
    df.show(5, False)
    df.printSchema()
    return df


def read_data(spark, input_path):
    rawDf = spark.read.format('csv').option('header', 'true').load(input_path)
    return rawDf

def create_sparksession():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('Iris decision tree pipelines').getOrCreate()
    return spark

def modelEvaluator(evaluator, predictedDf):
    print("calculate model evaluation metrics: ")

    rmse = evaluator.setMetricName("rmse").evaluate(predictedDf)
    mse = evaluator.setMetricName("mse").evaluate(predictedDf)
    r2 = evaluator.setMetricName("r2").evaluate(predictedDf)
    mae = evaluator.setMetricName("mae").evaluate(predictedDf)

    print(f"rmse = {rmse}, mse = {mse}, r2 = {r2}, mae = {mae}")

    return (rmse, mse, r2, mae)

def calcAllModelParamsAndMetrics(tvsModel):
    paramsAndMetrics = list(zip(tvsModel.validationMetrics, tvsModel.getEstimatorParamMaps()))

    for i in range(len(paramsAndMetrics)):
        print(paramsAndMetrics[i])
        print()

    return paramsAndMetrics

if __name__ == '__main__':
    run()
