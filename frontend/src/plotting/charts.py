"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(
    data: pd.DataFrame, col1: str, col2: str, title: str
) -> matplotlib.figure.Figure:
    """Построение heatmap"""

    fig = plt.figure(figsize=(15, 7))
    correlation = data[[col1, col2]].corr(method="spearman")
    sns.heatmap(correlation, annot=True)
    plt.title(title, fontsize=20)
    return fig


def boxplots(
    data: pd.DataFrame, x: str, y: str, title: str, ylim: int = None
) -> matplotlib.figure.Figure:
    """Создание графиков boxplot"""

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(x=x, y=y, data=data)

    plt.ylim(0, ylim)
    plt.title(title, fontsize=16)
    plt.ylabel(y, fontsize=14)
    plt.xlabel(x, fontsize=14)
    return fig
