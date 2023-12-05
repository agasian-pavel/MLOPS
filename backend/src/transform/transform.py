"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import warnings
import pandas as pd
from transliterate import translit
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {
        column: unique_df[column].explode().unique().tolist()
        if isinstance(unique_df[column].iloc[0], list)
        else unique_df[column].unique().tolist()
        for column in unique_df.columns
    }
    with open(unique_values_path, "w", encoding="utf-8") as file:
        json.dump(dict_unique, file, ensure_ascii=False)


def pipeline_preprocess(data: pd.DataFrame, **kwargs):
    """
    Обрабатывает исходные данные, применяя различные преобразования и заполнение отсутствующих значений.

    Аргументы:
    - data: pandas.DataFrame, исходные данные

    Возвращает:
    - pandas.DataFrame, обработанные данные
    """
    data = data.drop(kwargs["drop_columns"], axis=1, errors="ignore")
    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train

    data = data.drop(kwargs["drop_columns"], axis=1, errors="ignore")

    data = data.fillna(
        {
            col: -1000 if data[col].dtype in ["int64", "float64"] else "unknown"
            for col in data.columns
        }
    )

    # преобразование столбцов со списками в ячейках в бинаризованные
    list_columns = [col for col in data.columns if isinstance(data[col].iloc[0], list)]
    if len(list_columns) > 0:
        mlb = MultiLabelBinarizer()
        list_columns = [
            col for col in data.columns if isinstance(data[col].iloc[0], list)
        ]
        data_bin = pd.DataFrame(
            mlb.fit_transform(data[list_columns[0]]),
            columns=mlb.classes_,
            index=data.index,
        )
        data = pd.concat([data.drop(list_columns, axis=1), data_bin], axis=1)

    translit_columns = {
        column: translit(
            column.replace(" ", "_").replace("'", ""), "ru", reversed=True
        ).replace("'", "")
        for column in data.columns
    }
    data = data.rename(columns=translit_columns)

    data = transform_types(data=data, change_type_columns=kwargs["change_type_columns"])

    dict_category = {key: "category" for key in data.select_dtypes(["object"]).columns}
    data = transform_types(data=data, change_type_columns=dict_category)

    return data
