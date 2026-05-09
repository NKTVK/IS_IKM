import gradio as gr
import pickle
import pandas as pd
import numpy as np

# 1. Загружаем модель
with open('best_books_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Нам нужны те же средние и std, что были при обучении, чтобы правильно обработать ввод
# В идеале их надо было сохранить в файл, но для прототипа можно взять значения из анализа
# Либо быстро пересчитать их здесь из books.csv
df_stats = pd.read_csv('books.csv', on_bad_lines="skip")
df_stats.columns = [col.strip() for col in df_stats.columns]

# Словарик для языков (такой же как в train.py)
unique_languages = df_stats['language_code'].unique()
lang_map = {lang: i for i, lang in enumerate(unique_languages)}

# Параметры масштабирования
means = df_stats[['num_pages', 'ratings_count', 'text_reviews_count']].mean()
stds = df_stats[['num_pages', 'ratings_count', 'text_reviews_count']].std()


def predict_rating(num_pages, ratings_count, text_reviews_count, language):
    # А) Кодируем язык
    lang_id = lang_map.get(language, 0)

    # Б) Масштабируем числа (как в 8-й практике)
    pages_scaled = (num_pages - means['num_pages']) / stds['num_pages']
    ratings_scaled = (ratings_count - means['ratings_count']) / stds['ratings_count']
    reviews_scaled = (text_reviews_count - means['text_reviews_count']) / stds['text_reviews_count']

    # В) Собираем в таблицу (DataFrame), чтобы модель поняла
    input_data = pd.DataFrame([[pages_scaled, ratings_scaled, reviews_scaled, lang_id]],
                              columns=['num_pages', 'ratings_count', 'text_reviews_count', 'language_code'])

    # Г) Предсказание
    prediction = model.predict(input_data)[0]
    return f"Прогноз рейтинга: {prediction:.2f}"


# 2. Создаем интерфейс Gradio
interface = gr.Interface(
    fn=predict_rating,
    inputs=[
        gr.Number(label="Количество страниц", value=300),
        gr.Slider(0, 1000000, label="Количество оценок", step=1000),
        gr.Slider(0, 50000, label="Количество отзывов", step=100),
        gr.Dropdown(choices=list(lang_map.keys()), label="Язык книги", value="eng")
    ],
    outputs=gr.Text(label="Результат"),
    title="Сервис предсказания рейтинга книг",
    description="Введите параметры книги, чтобы узнать её потенциальный рейтинг (на основе модели Random Forest)."
)

# 3. Запуск
if __name__ == "__main__":
    interface.launch()