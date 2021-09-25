import pandas as pd
import numpy as np

def combine_street_region(df):
    df["city_street"] = df["street"].copy()
    df["city_street"].fillna(value="nan", inplace=True)
    df["city_street"] = df["city"] + "_" + df["city_street"]

    return df

feature_pairs = [("reform_mean_year_building_500", "reform_mean_year_building_1000"),
                 ("reform_mean_floor_count_500", "reform_mean_floor_count_1000"),
                 ("reform_house_population_500", "reform_house_population_1000")]

def fill_reforms_500_as_1000(df):
    for (feature_500_name, feature_1000_name) in feature_pairs:
        indices_to_fill = df[feature_500_name].isna() & ~df[feature_1000_name].isna()
        values = df.loc[indices_to_fill][feature_1000_name]
        df.loc[indices_to_fill, feature_500_name] = values

    return df
