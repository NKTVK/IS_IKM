import pickle
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Добавляем корень проекта в sys.path, чтобы импорт models и web_interface работал
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import RandomForestModel

#  Фикстура: обученная модель на синтетических данных

@pytest.fixture(scope="module")
def trained_model() -> RandomForestModel:
    """
    Обучает RandomForestModel на маленьком синтетическом датасете.
    Не требует наличия books.csv или best_books_model.pkl.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "num_pages":          rng.integers(50, 1000, n).astype(float),
        "ratings_count":      rng.integers(10, 100_000, n).astype(float),
        "text_reviews_count": rng.integers(0, 5_000, n).astype(float),
        "language_code":      rng.choice(["eng", "fre", "spa", "ger"], n),
    })
    y = pd.Series(rng.uniform(2.5, 5.0, n))

    m = RandomForestModel(n_estimators=10, max_depth=5)
    m.fit(X, y)
    return m


#  Тест 1: модель загружается и делает предсказание без ошибок

def test_model_predict_runs(trained_model: RandomForestModel):
    """
    Тест 1: Проверяет, что основная функция предсказания
    загружается и работает (не падает) на небольшом корректном примере.
    """
    result = trained_model.predict_one(
        num_pages=320,
        ratings_count=15_000,
        text_reviews_count=400,
        language_code="eng",
    )
    assert result is not None, "predict_one() вернул None"


#  Тест 2: результат имеет правильный формат

def test_model_predict_output_format(trained_model: RandomForestModel):
    """
    Тест 2: Проверяет, что функция предсказания возвращает ответ
    в правильном формате — одно вещественное число в диапазоне [0, 5].
    """
    result = trained_model.predict_one(
        num_pages=200,
        ratings_count=5_000,
        text_reviews_count=100,
        language_code="eng",
    )

    # Тип: float (или совместимый числовой тип numpy)
    assert isinstance(result, (float, np.floating)), (
        f"Ожидался float, получен {type(result)}"
    )

    # Рейтинг книг всегда в диапазоне от 0 до 5
    assert 0.0 <= result <= 5.0, (
        f"Рейтинг {result:.4f} вышел за допустимый диапазон [0, 5]"
    )

    # Не NaN и не бесконечность
    assert np.isfinite(result), f"Предсказание не является конечным числом: {result}"


#  Тест 3: веб-приложение инициализируется без ошибок

def test_web_interface_loads(trained_model: RandomForestModel, tmp_path, monkeypatch):
    """
    Тест 3: Проверяет, что веб-приложение
    может быть создано без ошибок.
    """
    import gradio as gr

    # Сохраняем временную модель в tmp_path
    model_path = tmp_path / "best_books_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)

    # Меняем рабочую директорию — open('best_books_model.pkl') найдёт файл
    monkeypatch.chdir(tmp_path)

    # Убираем web_interface из кэша, чтобы он переимпортировался заново
    monkeypatch.delitem(sys.modules, "web_interface", raising=False)

    # Импортируем модуль — это запускает весь код верхнего уровня,
    # включая pickle.load() и создание gr.Interface.
    import web_interface as wi

    # Объект интерфейса создан и является экземпляром Gradio
    assert isinstance(wi.interface, gr.Interface), (
        "web_interface.interface должен быть экземпляром gradio.Interface"
    )

    # Функция предсказания возвращает строку с нужным текстом
    output = wi.predict_rating(300, 10_000, 300, "eng")
    assert isinstance(output, str) and "Прогноз рейтинга" in output, (
        f"Неожиданный вывод интерфейса: {output!r}"
    )