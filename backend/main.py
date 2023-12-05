"""
Программа: Модель для прогнозирования стоимости аренды недвижимости в Москве под конкретные запросы клиента.
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class FlatParams(BaseModel):
    """
    Признаки для получения результатов модели
    """

    room_count: int
    district: str
    area: str
    metro: str
    time_metro: int
    floor: int
    attic: int
    square: int
    transport_type: str
    Vanna: int
    Dushevaja_kabina: int
    Internet: int
    Konditsioner: int
    Mebel_v_komnatah: int
    Mebel_na_kuhne: int
    Mozhno_s_detmi: int
    Mozhno_s_zhivotnymi: int
    Posudomoechnaja_mashina: int
    Stiralnaja_mashina: int
    Televizor: int
    Holodilnik: int


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(flat: FlatParams):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            flat.room_count,
            flat.district,
            flat.area,
            flat.metro,
            flat.time_metro,
            flat.floor,
            flat.attic,
            flat.square,
            flat.transport_type,
            flat.Vanna,
            flat.Dushevaja_kabina,
            flat.Internet,
            flat.Konditsioner,
            flat.Mebel_v_komnatah,
            flat.Mebel_na_kuhne,
            flat.Mozhno_s_detmi,
            flat.Mozhno_s_zhivotnymi,
            flat.Posudomoechnaja_mashina,
            flat.Stiralnaja_mashina,
            flat.Televizor,
            flat.Holodilnik,
        ]
    ]

    cols = [
        "room_count",
        "district",
        "area",
        "metro",
        "time_metro",
        "floor",
        "attic",
        "square",
        "transport_type",
        "Vanna",
        "Dushevaja_kabina",
        "Internet",
        "Konditsioner",
        "Mebel_v_komnatah",
        "Mebel_na_kuhne",
        "Mozhno_s_detmi",
        "Mozhno_s_zhivotnymi",
        "Posudomoechnaja_mashina",
        "Stiralnaja_mashina",
        "Televizor",
        "Holodilnik",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)

    return {"prediction": predictions}


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
