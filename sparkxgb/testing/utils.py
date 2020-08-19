from pyspark.sql.session import SparkSession


def default_session(conf=None):
    if conf is None:
        conf = {"spark.jars.packages": "ml.dmlc:xgboost4j_2.12:1.0.0,ml.dmlc:xgboost4j-spark_2.12:1.0.0"}

    builder = SparkSession.builder.appName("spark-xgboost")
    for key, value in conf.items():
        builder = builder.config(key, value)

    session = builder.getOrCreate()

    return session
