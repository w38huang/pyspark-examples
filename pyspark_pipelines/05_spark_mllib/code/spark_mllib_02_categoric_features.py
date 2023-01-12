import inspect

from pyspark.sql import SparkSession
from pyspark.sql.types import *

def spark_mllib_02_categoric_features():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    spark = SparkSession.builder.appName('spark_mllib_01_numeric_features').getOrCreate()

    superstore = spark.read.format("csv").option("header", "true").load("../datasets/superstore_data.csv")
    superstore.show()
    superstore.printSchema()


    ### EDA: ???
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol='Product Name', outputCol='Product Name Words')
    tokenized_data = tokenizer.transform(superstore.select('Product Name'))
    tokenized_data.select('Product Name', 'Product Name Words').show()

    # ## RegexTokenizer
    # from pyspark.ml.feature import RegexTokenizer
    # regexTokenizer = RegexTokenizer(inputCol='Product ID', outputCol='Product ID Tokens', pattern='-')
    # tokenized_data = regexTokenizer.transform(superstore.select('Product ID'))
    #
    # tokenized_data.select('Product ID', 'Product ID Tokens').show()
    # superstore.select('Category').distinct().show()

    ## StringIndexer
    from pyspark.ml.feature import StringIndexer
    indexer = StringIndexer(inputCol="Category", outputCol="CategoryIndex")
    superstore_catindex = indexer.fit(superstore).transform(superstore)
    superstore_catindex.select('Category', 'CategoryIndex').show()
    superstore.select('Ship Mode').distinct()

    from pyspark.ml.feature import OneHotEncoder

    oneHotEncoder = OneHotEncoder(inputCol="Ship Mode", outputCol="ShipMode_OneHotVector")
    # superstore_onehot = oneHotEncoder.fit(superstore).transform(superstore)
    shipmodeIndexer = StringIndexer(inputCol="Ship Mode", outputCol="ShipModeIndex")
    superstore_shipmodeindex = shipmodeIndexer.fit(superstore).transform(superstore)
    superstore_shipmodeindex.select('Ship Mode', 'ShipModeIndex').show()

    oneHotEncoder = OneHotEncoder(inputCol="ShipModeIndex", outputCol="ShipMode_OneHotVector")
    superstore_onehot = oneHotEncoder.fit(superstore_shipmodeindex).transform(superstore_shipmodeindex)
    superstore_onehot.select('Ship Mode', 'ShipModeIndex', 'ShipMode_OneHotVector').show()
    superstore.agg({'Postal Code': 'max'}).show()

    # oneHotEncoder_zipcode = OneHotEncoder(inputCol="Postal Code", outputCol="PostalCode_OneHotVector")
    # superstore_onehot = oneHotEncoder_zipcode.fit(superstore).transform(superstore)
    # superstore_onehot.select('Postal Code', 'PostalCode_OneHotVector').show()

    aaa = 1

def test():
    print(">>>> This is the begining of def {}().".format(inspect.stack()[0][3]))

    print("<<<< This is the end of def {}().".format(inspect.stack()[0][3]))


if __name__ == '__main__':
    # create_dataframe()
    spark_mllib_02_categoric_features()

# coding: utf-8



