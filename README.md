# spark-xgboost

![Build](https://github.com/sllynn/spark-xgboost/workflows/Build/badge.svg)

This fork updates the tests and release to use version 2.0.3 of xgboost4j and pyspark 3.5.0, also adds a makefile.

See original author repo at [https://github.com/sllynn/spark-xgboost](https://github.com/sllynn/spark-xgboost) for credits and historical context.

Spark users can use XGBoost for classification and regression tasks in a distributed environment through the excellent [XGBoost4J-Spark](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html) library, since this version has no official python bindings.

There is a current official implementation of [XGboost in pyspark](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.spark) that is Python specific, you probaly should use this one if your pipelines don't need models to have underlying JVM objects (e.g. to interact with Scala spark). 

For the current versions of [MLeap](https://github.com/combust/mleap) there is a need for an underlying JVM object see [this issue](https://github.com/combust/mleap/issues/867), so yhis wrapper allows to use the JVM Spark version in Python, making it compatible.

See the notebook in `/examples` for a practical illustration of usage.