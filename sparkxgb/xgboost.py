#
# Copyright (c) 2019 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict
from pyspark import keyword_only
from sparkxgb.common import XGboostEstimator, XGboostModel


class XGBoostClassifier(XGboostEstimator):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
    """

    @keyword_only
    def __init__(self, xgboost_params):
        super(XGBoostClassifier, self).__init__(
            classname="ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier"
        )
        self.setParams(xgboost_params=xgboost_params)

    @keyword_only
    def setParams(self, xgboost_params):
        kwargs = self._input_kwargs

        return self._set(**xgboost_params)

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model=java_model)


class XGBoostClassificationModel(XGboostModel):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
    """

    def __init__(
        self,
        classname="ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel",
        java_model=None,
    ):
        super(XGBoostClassificationModel, self).__init__(
            classname=classname, java_model=java_model
        )

    def getScore(self, feature: str = None, scoreType="weight") -> Dict[str, float]:

        jvm_map = self.nativeBooster.getScore(feature, scoreType)

        jvm_keys = jvm_map.keys().toList()

        keys = [jvm_keys.apply(i) for i in range(0, jvm_keys.length())]

        return {k: jvm_map.apply(k) for k in keys}

    def getNumFeature(self) -> int:

        return self.nativeBooster.getNumFeature()

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")


class XGBoostRegressor(XGboostEstimator):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
    """

    @keyword_only
    def __init__(self, xgboost_params):
        super(XGBoostRegressor, self).__init__(
            classname="ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor"
        )
        self.setParams(xgboost_params=xgboost_params)

    @keyword_only
    def setParams(self, xgboost_params):
        return self._set(**xgboost_params)

    def _create_model(self, java_model):
        return XGBoostRegressionModel(java_model=java_model)


class XGBoostRegressionModel(XGboostModel):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
    """

    def __init__(
        self,
        classname="ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel",
        java_model=None,
    ):
        super(XGBoostRegressionModel, self).__init__(
            classname=classname, java_model=java_model
        )

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")
