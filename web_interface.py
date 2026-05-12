import gradio as gr
import pickle

from models import RandomForestModel

#  1. Загружаем модель
with open('best_books_model.pkl', 'rb') as f:
    model: RandomForestModel = pickle.load(f)

# Список языков берём из маппинга, сохранённого внутри модели
available_languages = list(model.lang_map.keys())
default_lang = "eng" if "eng" in available_languages else available_languages[0]


#  2. Функция предсказания
def predict_rating(
    num_pages: float,
    ratings_count: float,
    text_reviews_count: float,
    language: str,
) -> str:
    prediction = model.predict_one(
        num_pages=num_pages,
        ratings_count=ratings_count,
        text_reviews_count=text_reviews_count,
        language_code=language,
    )
    return f"Прогноз рейтинга: {prediction:.2f} / 5.00"


#  3. Gradio-интерфейс
interface = gr.Interface(
    fn=predict_rating,
    inputs=[
        gr.Number(label="Количество страниц", value=300),
        gr.Slider(minimum=0, maximum=1_000_000, step=1_000, value=5_000,
                  label="Количество оценок"),
        gr.Slider(minimum=0, maximum=50_000, step=100, value=200,
                  label="Количество текстовых отзывов"),
        gr.Dropdown(choices=available_languages, value=default_lang,
                    label="Язык книги"),
    ],
    outputs=gr.Text(label="Результат"),
    title="Сервис предсказания рейтинга книг",
    description=(
        "Введите параметры книги, чтобы узнать её потенциальный рейтинг.\n"
        "Модель: RandomForestRegressor, обёрнутый в класс RandomForestModel."
    ),
    flagging_mode="never",
)

#  4. Запуск
if __name__ == "__main__":
    interface.launch(share=True)