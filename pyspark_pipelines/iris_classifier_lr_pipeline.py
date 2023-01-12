import inspect

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col

def run():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    input_path = 'data/common/iris/csv/iris.csv'

    # step 1: create a sparksession
    spark = create_sparksession()

    # step 2: read input dataframe
    rawDf = read_data(spark, input_path)

    # step 3: preprocessing
    df = preprocess(rawDf)

    # step 4: train test split
    (train, test) = df.randomSplit([0.7, 0.3], seed=100)

    # step5: build ml pipeline
    rawLabelColumn = "class"
    labelColumn = "classIndex"
    featuresColumn = "features"
    predictionColumn = "prediction"
    predictedLabel = "predictedLabel"

    labelIndexer, vectorAssembler, labelConverter = create_feature_pipeline(df, rawLabelColumn, labelColumn, featuresColumn, predictionColumn, predictedLabel)

    # step 6: choose a model
    model = LogisticRegression().setFeaturesCol(featuresColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)

    # step 7: create a model pipeline
    stages = [labelIndexer, vectorAssembler, model, labelConverter]
    pipeline = Pipeline().setStages(stages)

    # step 8: setup hyper parameter tuning parameters
    params = ParamGridBuilder().addGrid(model.regParam, [0.1, 0.5, 0.9]).addGrid(model.elasticNetParam, [0.2, 0.6, 0.8]).build()

    # step 9: setup evaluation metrics: f1
    evaluator = MulticlassClassificationEvaluator().setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    f1Evaluator = evaluator.setMetricName("f1")

    # step 10: train validation split
    tvs = (TrainValidationSplit()
           .setTrainRatio(0.75)
           .setSeed(123456)
           .setEstimatorParamMaps(params)
           .setEstimator(pipeline)
           .setEvaluator(f1Evaluator))

    tvsModel = tvs.fit(train)

    # step 11: get bestPipelineModel
    bestPipelineModel = tvsModel.bestModel

    # step 12: get bestModel
    bestModel = bestPipelineModel.stages[2]

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


def create_feature_pipeline(df, rawLabelColumn, labelColumn, featuresColumn, predictionColumn, predictedLabel):

    labelIndexer = StringIndexer().setInputCol(rawLabelColumn).setOutputCol(labelColumn).setHandleInvalid("skip")
    # inputCols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    inputCols = df.columns
    inputCols.remove(rawLabelColumn)

    vectorAssembler = VectorAssembler().setInputCols(inputCols).setOutputCol(featuresColumn)

    labels = labelIndexer.fit(df).labels
    labelConverter = IndexToString().setInputCol(predictionColumn).setOutputCol(predictedLabel).setLabels(labels)
    return (labelIndexer, vectorAssembler, labelConverter)

def preprocess(rawDf):
    df = (rawDf
        .withColumnRenamed("sepal.length", "sepal_length")
        .withColumnRenamed("sepal.width", "sepal_width")
        .withColumnRenamed("petal.length", "petal_length")
        .withColumnRenamed("petal.width", "petal_width")
        .withColumnRenamed("variety", "class")
        .select(
        col("sepal_length").cast("double"),
        col("sepal_width").cast("double"),
        col("petal_length").cast("double"),
        col("petal_width").cast("double"),
        col("class")
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

    f1 = evaluator.setMetricName("f1").evaluate(predictedDf)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictedDf)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictedDf)
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictedDf)

    print(f"f1 = {f1}, precicion = {precision}, recall = {recall}, accuracy = {accuracy}")

    return (f1, precision, recall, accuracy)

def calcAllModelParamsAndMetrics(tvsModel):
    paramsAndMetrics = list(zip(tvsModel.validationMetrics, tvsModel.getEstimatorParamMaps()))

    for i in range(len(paramsAndMetrics)):
        print(paramsAndMetrics[i])
        print()

    return paramsAndMetrics

if __name__ == '__main__':
    run()
