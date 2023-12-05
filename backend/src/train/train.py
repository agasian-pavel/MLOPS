"""
Программа: Тренировка данных
Версия: 1.0
"""

import optuna
from lightgbm import LGBMRegressor

from optuna import Study

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data, get_train_val_data
from ..train.metrics import save_metrics


def objective(
    trial,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_folds: int = 5,
    random_state: int = 10,
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    param_grid = {
        "objective": trial.suggest_categorical("objective", ["mae"]),
        "n_estimators": trial.suggest_categorical("n_estimators", [2749]),
        #         "learning_rate":
        #         trial.suggest_float("learning_rate", 0.01, 0.6, log=True),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.11790704731113073]
        ),
        "num_leaves": trial.suggest_int("num_leaves", 250, 2100),
        "max_depth": trial.suggest_int("max_depth", 8, 9),
        "reg_alpha": trial.suggest_int("reg_alpha", 30, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 80, 1000),
        "reg_lambda": trial.suggest_int("reg_lambda", 250, 2000),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.1, 0.9),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
        "metric": trial.suggest_categorical("metric", ["mae"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "early_stopping_rounds": trial.suggest_categorical(
            "early_stopping_rounds", [100]
        ),
        "verbose": trial.suggest_categorical("verbose", [-1]),
    }

    cv_folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_folds.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")
        model = LGBMRegressor(**param_grid)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="l1",
            callbacks=[pruning_callback],
        )

        predict = model.predict(x_test)
        cv_predicts[idx] = mean_absolute_error(y_test, predict)
    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMRegressor tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMRegressor
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )
    x_train_, x_val, y_train_, y_val = get_train_val_data(
        data_train=data_train, target=target
    )
    eval_set = [(x_val, y_val)]

    # training optimal params
    clf = LGBMRegressor(**study.best_params)
    clf.fit(x_train_, y_train_, eval_set=eval_set)
    # добавить eval_set

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf
