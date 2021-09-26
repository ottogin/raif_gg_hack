import pandas as pd
import numpy as np
from math import floor

train_floor_mapping = {
    'подвал, 1': '-1, 1', 
    'подвал': '-1', 
    'цоколь, 1': '0, 1', 
    '1,2,антресоль': '1, 2', 
    'цоколь': '0', 
    'тех.этаж (6)': '6, техэтаж', 
    'Подвал': '-1',
    'Цоколь': '0',
    'фактически на уровне 1 этажа': '1', 
    '1,2,3': '1, 2, 3', 
    '1, подвал': '-1, 1',
    '1,2,3,4': '1, 2, 3, 4', 
    '1,2': '1, 2', 
    '1,2,3,4,5': '1, 2, 3, 4, 5', 
    '5, мансарда': 'мансарда', 
    '1-й, подвал': '-1, 1', 
    '1, подвал, антресоль': '-1, 1',
    'мезонин': 'мезонин', 
    'подвал, 1-3': '-1, 1, 2, 3', 
    '1 (Цокольный этаж)': '0',
    '3, Мансарда (4 эт)': '3, мансарда', 
    'подвал,1': '-1, 1', 
    '1, антресоль': '1', 
    '1-3': '1, 2, 3',
    'мансарда (4эт)': 'мансарда', 
    '1, 2.': '1, 2', 
    'подвал , 1 ': '-1, 1', 
    '1, 2': '1, 2',
    'подвал, 1,2,3': '-1, 1, 2, 3', 
    '1 + подвал (без отделки)': '-1, 1', 
    'мансарда': 'мансарда',
    '2,3': '2, 3', 
    '4, 5': '4, 5', 
    '1-й, 2-й': '1, 2',
    '1 этаж, подвал': '-1, 1', 
    '1, цоколь': '0, 1', 
    'подвал, 1-7, техэтаж': '-1, 1, 2, 3, 4, 5, 6, 7, техэтаж', 
    '3 (антресоль)': '3', 
    '1, 2, 3': '1, 2, 3',
    'Цоколь, 1,2(мансарда)': '0, 1, мансарда', 
    'подвал, 3. 4 этаж': '-1, 3, 4', 
    'подвал, 1-4 этаж': '-1, 1, 2, 3, 4',
    'подва, 1.2 этаж': '-1, 1, 2', 
    '2, 3': '2, 3',
    '1.2': '1, 2', 
    '7,8': '7, 8',
    '1 этаж': '1', 
    '1-й': '1', 
    '3 этаж': '3', 
    '4 этаж': '4', 
    '5 этаж': '5', 
    'подвал,1,2,3,4,5': '-1, 1, 2, 3, 4, 5',
    'подвал, цоколь, 1 этаж': '-1, 0, 1', 
    '3, мансарда': '3, мансарда'
}


test_floor_mapping = {
    '2,3': '2, 3', 
    '1, 2': '1, 2',
    '1,2,3': '1, 2, 3', 
    '1,2,3,4': '1, 2, 3, 4', 
    'цоколь': '0', 
    'подвал': '-1', 
    'цоколь, 1, 2,3,4,5,6': '0, 1, 2, 3, 4, 5, 6',
    '1,2': '1, 2',
    ' 1, 2, Антресоль': '1, 2', 
    '3 этаж, мансарда (4 этаж)': '3, мансарда',
    'цокольный': '0', 
    '1-й, 2-й': '1, 2', 
    '1, подвал': '-1, 1', 
    '1, 2, 3': '1, 2, 3', 
    '1,2 ': '1, 2',
    'подвал,1': '-1, 1', 
    '1-й': '1', 
    '3,4': '3, 4', 
    'мансарда': 'мансарда',
    'подвал, 1 и 4 этаж': '-1, 1, 4',
    '5(мансарда)': 'мансарда',
    'технический этаж,5,6': '5, 6, техэтаж', 
    ' 1-2, подвальный': '-1, 1, 2', 
    '1, 2, 3, мансардный': '1, 2, 3, мансарда',
    'подвал, 1, 2, 3': '-1, 1, 2, 3',
    '1,2,3, антресоль, технический этаж': '1, 2, 3, техэтаж', 
    '3, 4': '3, 4', 
    '4, 5': '4, 5',
    '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)': '-1, 0, 1, 2, 3',
    '1,2,3,4, подвал': '-1',
    '2-й': '2', 
    '1, 2 этаж': '1, 2', 
    '1,2,3,4,5': '1, 2, 3, 4, 5', 
    'подвал, 1, 2': '-1, 1, 2',
    '1-7': '1, 2, 3, 4, 5, 6, 7',
    '1 (по док-м цоколь)': '1',
    '1-й, подвал': '-1, 1',
    '1,2,подвал ': '-1, 1, 2',
    'подвал, 2': '-1, 2', 
    '1, цоколь': '0, 1', 
    'подвал,1,2,3': '-1, 1, 2, 3',
    '1,2,3 этаж, подвал': '-1, 1, 2, 3', 
    'цоколь, 1': '0, 1', 
    '2, 3, 4, тех.этаж': '2, 3, 4',
    'цокольный, 1,2': '0, 1, 2',
    'Техническое подполье': 'техэтаж',
}


def process_floor(df, floor_mapping):
    df['floor'] = df['floor'].astype(str)
    clean_floor = lambda x: floor_mapping[x] if x in floor_mapping.keys() else x
    df['floor'] = df['floor'].map(clean_floor) 
    float_floors = df['floor'].str.contains('\.')
    df.loc[float_floors, 'floor'] = df[float_floors]['floor'].map(lambda x: str(int(float(x))))
    return df


def create_bow(train):
    unique_floors = []
    for floor in train.floor.unique():
        unique_floors += floor.split(',')
    for i in range(len(unique_floors)):
        unique_floors[i] = unique_floors[i].strip()
    unique_floors = list(set(unique_floors))
    return unique_floors


def prepare_one_hot_floor(df, unique_floors):
    for floor in unique_floors:
        df['floor == {}'.format(floor)] = df.floor.str.contains(floor).astype(int)
    return df


def get_one_hot_floor_features(train_, test_):
    train, test = train_.copy(), test_.copy()
    train = process_floor(train, train_floor_mapping)
    test = process_floor(test, test_floor_mapping)
    unique_floors = create_bow(train)
    train = prepare_one_hot_floor(train, unique_floors)
    test = prepare_one_hot_floor(test, unique_floors)
    return train, test


# Nb and height features


def compute_nb_floors(df):
    df['nb_floors'] = df['floor'].map(lambda x: len(x.split(',')))
    return df


def compute_mean_height(df):
    def mean_height(x):
        x = x.split(',')
        for i in range(len(x)):
            x[i] = x[i].strip() 
            try:
                x[i] = int(x[i])
            except:
                x[i] = np.nan
        x = pd.Series(x)
        if x.notnull().sum() > 0:
            return x.mean()
        else:
            return np.nan
    df['floor_mean_height'] = df['floor'].map(mean_height)    
    return df


def get_floor_nb_and_height_features(train_, test_):
    train, test = train_.copy(), test_.copy()

    train = process_floor(train, train_floor_mapping)
    test = process_floor(test, test_floor_mapping)

    train = compute_nb_floors(train)
    test = compute_nb_floors(test)

    train = compute_mean_height(train)
    test = compute_mean_height(test)
    
    fillna = train['floor_mean_height'].median()
    train['floor_mean_height'] = train['floor_mean_height'].fillna(fillna)
    test['floor_mean_height'] = test['floor_mean_height'].fillna(fillna)
    fillna_one = train.groupby('region').floor_mean_height.mean()
    
    train_mask = train['floor_mean_height'].isna()
    test_mask = test['floor_mean_height'].isna()
    
    train.loc[train_mask, 'floor_mean_height'] = train[train_mask]['realty_type'].map(fillna_one)
    test.loc[test_mask, 'floor_mean_height'] = test[test_mask]['realty_type'].map(fillna_one)    
    
    quantile = train['floor_mean_height'].quantile(0.99)
    train['floor_mean_height'] = train['floor_mean_height'].clip(upper=quantile)
    test['floor_mean_height'] = test['floor_mean_height'].clip(upper=quantile)
    
    return train, test
