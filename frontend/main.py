"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.train.training import start_training
from src.plotting.charts import plot_correlation_heatmap, boxplots
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://i.postimg.cc/PTJZkFTr/Untitled-1-upscayl-4x-ultramix-balanced.jpg",
        width=800,
    )

    st.title(
        "MLOps project:  Расчет стоимости аренды квартиры в Москве на основе данных с сайта cian.ru"
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
            target - price - стоимость аренды
            room_count - количество комнат в квартире (0 - квартира-студия)
            district - округ Москвы
            area - район
            street - улица
            house - номер дома
            metro - ближайшая станция метро
            time_metro - время, которое необходимо затратить, чтобы добраться до ближайшей станции метро
            transport_type - способ передвижения до метро (пешком или на транспорте)
            facilities - удобства (кондиционер, мебель и т. д.) и условия проживания (можно с животными, детьми и т.д.)
            floor - этаж, на котором расположена квартира
            attic - количество этажей в доме
            square - площадь квартиры
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(config["preprocessing"]["train_path"])
    st.write(data.head())

    # plotting with checkbox
    attic_price = st.sidebar.checkbox("Количество этажей-Цена аренды")
    floor_price = st.sidebar.checkbox("Высота этажа-Цена аренды")
    transport_type_price = st.sidebar.checkbox("Дорога до метро-Цена аренды")

    if attic_price:
        st.write(
            "<p style='font-weight: bold;'>Гипотеза 1. Чем больше этажей в доме, тем дороже аренда.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            plot_correlation_heatmap(
                data=data,
                col1="attic",
                col2="price",
                title="Количество этажей-Цена аренды",
            )
        )
        st.write(
            "<p style='font-weight: bold;'>Корреляция между количеством этажей и стоимостью аренды отсутствует. Гипотеза опровергнута.</p>",
            unsafe_allow_html=True,
        )
    if floor_price:
        st.write(
            "<p style='font-weight: bold;'>Гипотеза 2. Чем выше этаж, тем дороже аренда.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            plot_correlation_heatmap(
                data=data,
                col1="floor",
                col2="price",
                title="Высота этажа-Цена аренды",
            )
        )
        st.write(
            "<p style='font-weight: bold;'>Корреляция отсутствует. Гипотеза опровергнута.</p>",
            unsafe_allow_html=True,
        )
    if transport_type_price:
        st.write(
            "<p style='font-weight: bold;'>Гипотеза 3. Квартиры в шаговой доступности от метро дороже, чем те, от которых необходимо добираться на транспорте.</p>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            boxplots(
                data=data,
                x="transport_type",
                y="price",
                title="Дорога до метро-Цена аренды",
                ylim=200000,
            )
        )
        st.write(
            "<p style='font-weight: bold;'>Медианная цена квартиры в шаговой доступности дороже, чем та, до которой необходимо добираться на транспорте. Гипотеза подтверждена</p>",
            unsafe_allow_html=True,
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Обучение модели LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Предсказание")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]
    all_data_path = config["preprocessing"]["all_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(
            unique_data_path=unique_data_path,
            all_data_path=all_data_path,
            endpoint=endpoint,
        )
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "json"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Разведочный анализ данных": exploratory,
        "Обучение модели": training,
        "Предсказание по параметрам": prediction,
        "Предсказание по параметрам из файла": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
