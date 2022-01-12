import pyspark
from pyspark.sql.session import SparkSession

SPARK_SCALA_MAPPING = {
    "2": "2.11",
    "3": "2.12"
}

def default_session(conf=None):
    spark_major_version = pyspark.__version__[0]
    scala_version = SPARK_SCALA_MAPPING[spark_major_version]
    mvn_group = "ml.dmlc"
    xgb_version = "1.3.1"
    xgboost4j_coords = f"{mvn_group}:xgboost4j_{scala_version}:{xgb_version}"
    xgboost4j_spark_coords = f"{mvn_group}:xgboost4j-spark_{scala_version}:{xgb_version}"

    if conf is None:
        conf = {
            "spark.jars.packages": ",".join([xgboost4j_coords, xgboost4j_spark_coords])
        }

    builder = SparkSession.builder.appName("spark-xgboost")
    for key, value in conf.items():
        builder = builder.config(key, value)

    session = builder.getOrCreate()

    return session
