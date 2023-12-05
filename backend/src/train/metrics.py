"""
Программа: Получение метрик
Версия: 1.0
"""
import json

import numpy as np
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)
import pandas as pd


def create_dict_metrics(y_test: pd.Series, y_predict: pd.Series) -> dict:
    """
    Получение словаря с метриками для задачи регрессии и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :return: словарь с метриками
    """
    dict_metrics = {
        "MAE": round(mean_absolute_error(y_test, y_predict), 3),
        "MSE": round(mean_squared_error(y_test, y_predict), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_predict)), 3),
        "R2": round(r2_score(y_test, y_predict), 3),
        "MAPE": round(mean_absolute_percentage_error(y_test, y_predict), 3),
        "MSLE": round(mean_squared_log_error(y_test, y_predict), 3),
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=data_y,
        y_predict=model.predict(data_x),
    )
    with open(metric_path, "w", encoding="utf-8") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
