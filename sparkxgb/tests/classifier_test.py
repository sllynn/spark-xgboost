import unittest

from pyspark.sql.types import StringType

from sparkxgb.xgboost import XGBoostClassifier, XGBoostClassificationModel
from sparkxgb.testing.utils import default_session
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class XGBClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spark = default_session()

        col_names = [
            "age", "workclass", "fnlwgt",
            "education", "education-num",
            "marital-status", "occupation",
            "relationship", "race", "sex",
            "capital-gain", "capital-loss",
            "hours-per-week", "native-country",
            "label"
        ]

        sdf = (
            self.spark.read
                .csv(path="./sparkxgb/tests/data/adult.data", inferSchema=True)
                .toDF(*col_names)
                .repartition(200)
        )

        string_columns = [fld.name for fld in sdf.schema.fields if isinstance(fld.dataType, StringType)]
        string_col_replacements = [fld + "_ix" for fld in string_columns]
        string_column_map = list(zip(string_columns, string_col_replacements))
        target = string_col_replacements[-1]
        predictors = [fld.name for fld in sdf.schema.fields if
                      not isinstance(fld.dataType, StringType)] + string_col_replacements[:-1]

        si = [StringIndexer(inputCol=fld[0], outputCol=fld[1]) for fld in string_column_map]
        va = VectorAssembler(inputCols=predictors, outputCol="features")
        pipeline = Pipeline(stages=[*si, va])
        fitted_pipeline = pipeline.fit(sdf)

        sdf_prepared = fitted_pipeline.transform(sdf)

        self.train_sdf, self.test_sdf = sdf_prepared.randomSplit([0.8, 0.2], seed=1337)

    def test_binary_classifier_args(self):

        self.spark.sparkContext.setLogLevel("INFO")

        xgb_params = dict(
            eta=0.1,
            maxDepth=2,
            missing=0.0,
            objective="binary:logistic",
            numRound=5,
            numWorkers=2,
            killSparkContextOnWorkerFailure=False
        )

        xgb = (
            XGBoostClassifier(**xgb_params)
            .setFeaturesCol("features")
            .setLabelCol("label_ix")
        )
        self.assertIsInstance(xgb, XGBoostClassifier)

    def test_binary_classifier(self):

        self.spark.sparkContext.setLogLevel("INFO")

        xgb_params = dict(
            eta=0.1,
            maxDepth=2,
            missing=0.0,
            objective="binary:logistic",
            numRound=5,
            numWorkers=2
        )

        xgb = (
            XGBoostClassifier(**xgb_params)
            .setFeaturesCol("features")
            .setLabelCol("label_ix")
        )

        bce = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label_ix"
        )

        model = xgb.fit(self.train_sdf)
        roc = bce.evaluate(model.transform(self.test_sdf))

        print(roc)

        self.assertIsInstance(model, XGBoostClassificationModel)
        self.assertGreater(roc, 0.8)


if __name__ == '__main__':
    unittest.main()
