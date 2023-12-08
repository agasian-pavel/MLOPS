"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
import locale


def evaluate_input(unique_data_path: str, all_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param all_data_path: путь до всех значений метро, районов и округов
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path, "r", encoding="utf-8") as file:
        unique_df = json.load(file)

    with open(all_data_path, "r", encoding="utf-8") as file:
        all_df = json.load(file)

    # поля для ввода данных, используем уникальные значения
    room_count = st.sidebar.selectbox(
        "Количество комнат",
        sorted(unique_df["room_count"]),
        format_func=lambda x: "Студия" if x == 0 else x,
    )
    metro = st.sidebar.selectbox(
        "Станция метро",
        sorted(unique_df["metro"], key=lambda x: str(x) if x is not None else ""),
    )
    if metro:
        area = st.sidebar.selectbox(
            "Район",
            list(all_df[metro].keys()),
        )
        district = st.sidebar.selectbox(
            "Округ",
            list(all_df[metro].values()),
        )
    time_metro = st.sidebar.slider(
        "Время до ближайшей станции метро",
        min_value=min((unique_df["time_metro"])),
        max_value=max((unique_df["time_metro"])),
    )
    attic = st.sidebar.slider(
        "Количество этажей в доме",
        min_value=min(unique_df["attic"]) + 1,
        max_value=max(unique_df["attic"]),
    )
    floor = st.sidebar.slider(
        "Этаж", min_value=min((unique_df["floor"])), max_value=attic
    )
    square = st.sidebar.slider(
        "Площадь квартиры",
        min_value=min((unique_df["square"])),
        max_value=max((unique_df["square"])),
    )
    transport_type = st.sidebar.selectbox(
        "Способ передвижения до метро", (unique_df["transport_type"])
    )
    fridge = st.sidebar.selectbox(
        "Холодильник", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    kids_option = st.sidebar.selectbox(
        "Можно с детьми", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Да"
    )
    pets_option = st.sidebar.selectbox(
        "Можно с животными", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Да"
    )
    kitch_furniture = st.sidebar.selectbox(
        "Мебель на кухне", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    room_furniture = st.sidebar.selectbox(
        "Мебель в комнатах", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    internet = st.sidebar.selectbox(
        "Интернет", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    wash_mach = st.sidebar.selectbox(
        "Стиральная машина", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    bathtub = st.sidebar.selectbox(
        "Ванная", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    tv = st.sidebar.selectbox(
        "Телевизор", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    air_cond = st.sidebar.selectbox(
        "Кондиционер", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    sh_cab = st.sidebar.selectbox(
        "Душевая", (0, 1), format_func=lambda x: "Нет" if x == 0 else "Есть"
    )
    dishwasher = st.sidebar.selectbox(
        "Посудомоечная машина",
        (0, 1),
        format_func=lambda x: "Нет" if x == 0 else "Есть",
    )

    dict_data = {
        "room_count": room_count,
        "metro": metro,
        "area": area,
        "district": district,
        "time_metro": time_metro,
        "attic": attic,
        "floor": floor,
        "square": square,
        "transport_type": transport_type,
        "Vanna": bathtub,
        "Dushevaja_kabina": sh_cab,
        "Internet": internet,
        "Konditsioner": air_cond,
        "Mebel_v_komnatah": room_furniture,
        "Mebel_na_kuhne": kitch_furniture,
        "Mozhno_s_detmi": kids_option,
        "Mozhno_s_zhivotnymi": pets_option,
        "Posudomoechnaja_mashina": dishwasher,
        "Stiralnaja_mashina": wash_mach,
        "Televizor": tv,
        "Holodilnik": fridge,
    }

    st.write(
        f"""### Данные для запроса:\n
        1. Количество комнат: {dict_data['room_count']}
        2. Станция метро: {dict_data['metro']}        
        3. Район: {dict_data['area']}
        4. Округ: {dict_data['district']}
        5. Время до ближайшей станции метро: {dict_data['time_metro']}
        6. Этаж: {dict_data['floor']}
        7. Количество этажей в доме: {dict_data['attic']}
        8. Площадь: {dict_data['square']}
        9. Способ передвижения до метро: {dict_data['transport_type']}
        10. Ванна: {dict_data['Vanna']}
        11. Душевая: {dict_data['Dushevaja_kabina']}
        12. Интернет: {dict_data['Internet']}
        13. Кондиционер: {dict_data['Konditsioner']}
        14. Мебель в комнатах: {dict_data['Mebel_v_komnatah']}
        15. Мебель на кухне: {dict_data['Mebel_na_kuhne']}
        16. Можно с детьми: {dict_data['Mozhno_s_detmi']}
        17. Можно с животными: {dict_data['Mozhno_s_zhivotnymi']}
        18. Посудомоечная машина: {dict_data['Posudomoechnaja_mashina']}
        19. Стиральная машина: {dict_data['Stiralnaja_mashina']}
        20. Телевизор: {dict_data['Televizor']}
        21. Холодильник: {dict_data['Holodilnik']}
        """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        locale.setlocale(locale.LC_ALL, "")  # Установка локали для форматирования чисел
        rent_cost = round(output["prediction"][0])
        formatted_rent_cost = locale.format_string(
            "%d", rent_cost, grouping=True
        )  # Форматирование числа с разделителем точкой
        st.write(f"## Стоимость аренды: {formatted_rent_cost} ₽")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())
