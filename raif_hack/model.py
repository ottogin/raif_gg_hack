import typing
import pickle
import pandas as pd
import numpy as np
import logging

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from raif_hack.data_transformers import SmoothedTargetEncoding

from raif_hack.settings import (
    MODEL_PARAMS,
    LOGGING_CONFIG,
    NUM_FEATURES,
    CATEGORICAL_OHE_FEATURES,
    CATEGORICAL_STE_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


class BenchmarkModel:
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(
        self,
        numerical_features: typing.List[str],
        ohe_categorical_features: typing.List[str],
        ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
        model_params: typing.Dict[str, typing.Union[str, int, float]],
    ):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.ste_cat_features,
                ),
            ]
        )

        self.model = LGBMRegressor(**model_params)

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", self.model)]
        )

        self._is_fitted = False
        self.corr_coef = 0

    def _find_corr_coefficient(self, X_manual: pd.DataFrame, y_manual: pd.Series):
        """Вычисление корректирующего коэффициента

        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        predictions = self.pipeline.predict(X_manual)
        deviation = ((y_manual - predictions) / predictions).median()
        self.corr_coef = deviation

    def fit(
        self,
        X_offer: pd.DataFrame,
        y_offer: pd.Series,
        X_manual: pd.DataFrame,
        y_manual: pd.Series,
    ):
        """Обучение модели.
        ML модель обучается на данных по предложениям на рынке (цены из объявления)
        Затем вычисляется среднее отклонение между руяными оценками и предиктами для корректировки стоимости

        :param X_offer: pd.DataFrame с объявлениями
        :param y_offer: pd.Series - цена предложения (в объявлениях)
        :param X_manual: pd.DataFrame с ручными оценками
        :param y_manual: pd.Series - цены ручника
        """
        logger.info("Fit lightgbm")
        self.pipeline.fit(
            X_offer,
            y_offer,
            model__feature_name=[f"{i}" for i in range(70)],
            model__categorical_feature=["67", "68", "69"],
        )
        logger.info("Find corr coefficient")
        self._find_corr_coefficient(X_manual, y_manual)
        logger.info(f"Corr coef: {self.corr_coef:.2f}")
        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            predictions = self.pipeline.predict(X)
            corrected_price = predictions * (1 + self.corr_coef)
            return corrected_price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )

    def save(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str):
        """Сериализует модель в pickle.

        :param path: str, путь до файла
        :return: Модель
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class TwoStepBenchmarkModel(BenchmarkModel):
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(
        self,
        numerical_features: typing.List[str],
        ohe_categorical_features: typing.List[str],
        ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
        model_params: typing.Dict[str, typing.Union[str, int, float]],
    ):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.ste_cat_features,
                ),
            ]
        )
        self.preprocessor2 = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features + ["killer_f"]),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.ste_cat_features,
                ),
            ]
        )

        # params =

        # logger.info("Init with ")

        self.model = CatBoostRegressor()
        self.model2 = CatBoostRegressor()

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", self.model)]
        )

        self.pipeline2 = Pipeline(
            steps=[("preprocessor", self.preprocessor2), ("model", self.model2)]
        )

        self._is_fitted = False
        self.corr_coef = 0

    def fit(
        self,
        X_offer: pd.DataFrame,
        y_offer: pd.Series,
        X_manual: pd.DataFrame,
        y_manual: pd.Series,
    ):

        logger.info("Fit catboost")
        self.pipeline.fit(
            X_offer,
            y_offer,
            # model__feature_name=NUM_FEATURES
            # + CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
            # model__categorical_feature=CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
        )

        killer_f = self.pipeline.predict(X_manual)
        X_manual = X_manual.copy()
        X_manual["killer_f"] = killer_f
        logger.info("Fit catboost 2")
        self.pipeline2.fit(
            X_manual,
            y_manual,
            # model__feature_name=NUM_FEATURES
            # + CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES
            # + ["killer_f"],
            # model__categorical_feature=CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
        )

        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            killer_f = self.pipeline.predict(X)
            X = X.copy()
            X["killer_f"] = killer_f
            price = self.pipeline2.predict(X)
            return price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )


class WeightedTwoStepModel(BenchmarkModel):
    """
    Модель представляет из себя sklearn pipeline. Пошаговый алгоритм:
      1) в качестве обучения выбираются все данные с price_type=0
      1) все фичи делятся на три типа (numerical_features, ohe_categorical_features, ste_categorical_features):
          1.1) numerical_features - применяется StandardScaler
          1.2) ohe_categorical_featires - кодируются через one hot encoding
          1.3) ste_categorical_features - кодируются через SmoothedTargetEncoder
      2) после этого все полученные фичи конкатенируются в одно пространство фичей и подаются на вход модели Lightgbm
      3) делаем предикт на данных с price_type=1, считаем среднее отклонение реальных значений от предикта. Вычитаем это отклонение на финальном шаге (чтобы сместить отклонение к 0)

    :param numerical_features: list, список численных признаков из датафрейма
    :param ohe_categorical_features: list, список категориальных признаков для one hot encoding
    :param ste_categorical_features, list, список категориальных признаков для smoothed target encoding.
                                     Можно кодировать сразу несколько полей (например объединять категориальные признаки)
    :
    """

    def __init__(
        self,
        numerical_features: typing.List[str],
        ohe_categorical_features: typing.List[str],
        ste_categorical_features: typing.List[typing.Union[str, typing.List[str]]],
        model_params: typing.Dict[str, typing.Union[str, int, float]],
    ):
        self.num_features = numerical_features
        self.ohe_cat_features = ohe_categorical_features
        self.ste_cat_features = ste_categorical_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.ste_cat_features,
                ),
            ]
        )
        self.preprocessor2 = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features + ["killer_f"]),
                ("ohe", OneHotEncoder(), self.ohe_cat_features),
                (
                    "ste",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.ste_cat_features,
                ),
            ]
        )

        # self.model = LGBMRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.01,
        #     reg_alpha=1,
        #     num_leaves=40,
        #     min_child_samples=5,
        #     importance_type="gain",
        #     n_jobs=4,
        #     random_state=563,
        # )
        # self.model2 = LGBMRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.01,
        #     reg_alpha=1,
        #     num_leaves=40,
        #     min_child_samples=5,
        #     importance_type="gain",
        #     n_jobs=4,
        #     random_state=213,
        # )

        self.model = CatBoostRegressor(loss_function="MAE", iterations=3000)
        self.model2 = CatBoostRegressor(loss_function="MAE", iterations=2000)

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", self.model)]
        )

        self.pipeline2 = Pipeline(
            steps=[("preprocessor", self.preprocessor2), ("model", self.model2)]
        )

        self._is_fitted = False
        self.corr_coef = 0

    def fit(
        self,
        X_offer: pd.DataFrame,
        y_offer: pd.Series,
        X_manual: pd.DataFrame,
        y_manual: pd.Series,
        X_val_offer: typing.Optional[pd.DataFrame] = None,
        y_val_offer: typing.Optional[pd.Series] = None,
        X_val_manual: typing.Optional[pd.DataFrame] = None,
        y_val_manual: typing.Optional[pd.Series] = None,
    ):

        logger.info("Fit catboost")

        X = pd.concat([X_offer, X_manual])
        y = pd.concat([y_offer, y_manual])
        WEIGHT = 0.05
        weight = np.ones_like(y.values) * WEIGHT
        weight[-len(y_manual) :] = 1 - WEIGHT

        X_prep = self.pipeline[:-1].fit_transform(
            X,
            y,
        )
        X_val = pd.concat([X_val_offer, X_val_manual])
        y_val = pd.concat([y_val_offer, y_val_manual])
        X_val_prep = self.pipeline[:-1].transform(X_val)
        self.pipeline[-1:].fit(
            X_prep, 
            y,
            # model__feature_name=NUM_FEATURES
            # + CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
            # model__categorical_feature=CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
            model__use_best_model=True,
            model__eval_set=Pool(X_val_prep, y_val),
            model__sample_weight=weight,
        )

        killer_f = self.pipeline.predict(X_manual)
        killer_f_val = self.pipeline.predict(X_val_manual)
        X_manual = X_manual.copy()
        X_val_manual = X_val_manual.copy()
        X_manual["killer_f"] = killer_f
        X_val_manual["killer_f"] = killer_f_val

        logger.info("Fit catboost 2")

        X_manual_prep = self.pipeline2[:-1].fit_transform(X_manual, y_manual)
        X_val_manual_prep = self.pipeline2[:-1].transform(X_val_manual)
        self.pipeline2[-1:].fit(
            X_manual_prep,
            y_manual,
            # model__feature_name=NUM_FEATURES
            # + CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES
            # + ["killer_f"],
            # model__categorical_feature=CATEGORICAL_OHE_FEATURES
            # + CATEGORICAL_STE_FEATURES,
            model__use_best_model=True,
            model__eval_set=Pool(X_val_manual_prep, y_val_manual)
        )

        self.__is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.array:
        """Предсказание модели Предсказываем преобразованный таргет, затем конвертируем в обычную цену через обратное
        преобразование.

        :param X: pd.DataFrame
        :return: np.array, предсказания (цены на коммерческую недвижимость)
        """
        if self.__is_fitted:
            killer_f = self.pipeline.predict(X)
            X = X.copy()
            X["killer_f"] = killer_f
            price = self.pipeline2.predict(X)
            return price
        else:
            raise NotFittedError(
                "This {} instance is not fitted yet! Call 'fit' with appropriate arguments before predict".format(
                    type(self).__name__
                )
            )
