import argparse
import logging.config
import pandas as pd
import numpy as np
from traceback import format_exc

from raif_hack.model import *
from raif_hack.settings import (
    MODEL_PARAMS,
    LOGGING_CONFIG,
    NUM_FEATURES,
    CATEGORICAL_OHE_FEATURES,
    CATEGORICAL_STE_FEATURES,
    TARGET,
)
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat
from raif_hack.features import prepare_categorical
from raif_hack.floor_processing import get_floor_nb_and_height_features
from raif_hack.streets_reforms_processing import combine_street_region, fill_reforms_500_as_1000

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--val", action="store_true")

    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        dest="mp",
        required=True,
        help="Куда сохранить обученную ML модель",
    )

    return parser.parse_args()


if __name__ == "__main__":

    try:
        logger.info("START train.py with")
        args = vars(parse_args())

        train_path = "data/train.csv"
        if args["val"]:
            train_path = "data/train_trunc.csv"

        logger.info("Load train df from %s" % train_path)
        train_df = pd.read_csv(train_path)
        logger.info(f"Input shape: {train_df.shape}")
        val_df = pd.read_csv("data/validation.csv")

        train_df, val_df = get_floor_nb_and_height_features(train_df, val_df)

        # Street encoding
        train_df = combine_street_region(train_df)
        val_df = combine_street_region(val_df)

        # Reform nan fill
        # train_df = fill_reforms_500_as_1000(train_df)
        # val_df = fill_reforms_500_as_1000(val_df)

        train_df = prepare_categorical(train_df)
        val_df = prepare_categorical(val_df)

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
        ]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
        ]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(
            f"X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}"
        )
        model = WeightedBlendComplex(
            numerical_features=NUM_FEATURES,
            ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
            ste_categorical_features=CATEGORICAL_STE_FEATURES,
            model_params=MODEL_PARAMS,
        )
        logger.info("Fit model")

        X_manual_val = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
        ]
        y_manual_val = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        X_offer_val = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][
            NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES
        ]
        y_offer_val = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

        model.fit(
            X_offer, y_offer, X_manual, y_manual, X_offer_val, y_offer_val, X_manual_val, y_manual_val,
            use_best_model=args["val"]
            )
        logger.info("Save model")
        model.save(args["mp"])

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f"Metrics stat for training data with manual prices: {metrics}")

        # Validation
        predictions_manual_val = model.predict(X_manual_val)
        metrics = metrics_stat(y_manual_val.values, predictions_manual_val)
        logger.info(f"Metrics stat for validation data with manual prices: {metrics}")

    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)
    logger.info("END train.py")
