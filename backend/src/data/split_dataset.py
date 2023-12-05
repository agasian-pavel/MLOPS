"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Разделение данных на train/test с последующим сохранением
    :param dataset: датасет
    :return: train/test датасеты
    """
    # Split in train/test
    df_train, df_test = train_test_split(
        dataset,
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
    )
    df_train.to_json(kwargs["train_path_proc"], orient="records")
    df_test.to_json(kwargs["test_path_proc"], orient="records")
    return df_train, df_test


def get_train_test_data(
    data_train: pd.DataFrame, data_test: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param data_test: test датасет
    :param target: название целевой переменной
    :return: набор данных train/test
    """
    x_train, x_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )
    return x_train, x_test, y_train, y_test


def get_train_val_data(
    data_train: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/val данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param target: название целевой переменной
    :return: набор данных train/validation
    """
    train_size = len(data_train)
    val_size = int(0.16 * train_size)
    train_indices = list(range(train_size - val_size))
    val_indices = list(range(train_size - val_size, train_size))

    x_train_, x_val = (
        data_train.drop(target, axis=1).iloc[train_indices],
        data_train.drop(target, axis=1).iloc[val_indices],
    )
    y_train_, y_val = (
        data_train.loc[:, target].iloc[train_indices],
        data_train.loc[:, target].iloc[val_indices],
    )

    return x_train_, x_val, y_train_, y_val
