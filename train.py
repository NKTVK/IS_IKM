import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from models import SimpleModel, RandomForestModel

#  1. Подготовка данных
print("1. Подготовка данных")

# Загрузка файла + пропуск грязных строк
df = pd.read_csv('books.csv', on_bad_lines="skip")
# Очистка от лишних пробелов в названиях столбцов
df.columns = [col.strip() for col in df.columns]

# Признаки (X): num_pages (кол-во страниц), ratings_count (кол-во оценок),
#               text_reviews_count (кол-во отзывов), language_code (язык книги)
# Правильный ответ (y): average_rating (средняя оценка книги)
features = ['num_pages', 'ratings_count', 'text_reviews_count', 'language_code']
target = 'average_rating'

X = df[features].copy()
y = df[target].copy()

# Сохраняем X_raw с оригинальными значениями для RandomForestModel —
# она выполняет кодирование языка и масштабирование самостоятельно внутри себя.
X_raw = X.copy()

# Превращаем категориальный признак language_code в числа простым сопоставлением словарю.
# Эта закодированная версия (X) используется только для SimpleModel.
unique_languages = X['language_code'].unique()
lang_map = {lang: i for i, lang in enumerate(unique_languages)}
X['language_code'] = X['language_code'].map(lang_map)

# Масштабирование числовых признаков по формуле (x - mean) / std:
# вычитаем среднее и делим на стандартное отклонение.
# Применяется только к X для SimpleModel.
# X_raw намеренно НЕ масштабируется здесь — RandomForestModel сделает это сама.
numeric_cols = ['num_pages', 'ratings_count', 'text_reviews_count']
for col in numeric_cols:
    mean_val = X[col].mean()
    std_val  = X[col].std()
    X[col] = (X[col] - mean_val) / std_val

# Перемешиваем данные для воспроизводимости
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
X     = X.iloc[indices]
X_raw = X_raw.iloc[indices]
y     = y.iloc[indices]

# Разделяем 80% на обучающую/валидационную и 20% на итоговый тест.
# Соотношение 80/20 — хороший выбор для датасета среднего размера (как у нас):
# 80% достаточно для качественного обучения, а 20% даёт репрезентативную оценку.
# При 75/25 потеря для обучения заметнее, а выигрыш на тесте незначителен.
split_idx = int(len(X) * 0.8)

X_train_val,     X_final_test     = X[:split_idx],     X[split_idx:]
X_raw_train_val, X_raw_final_test = X_raw[:split_idx], X_raw[split_idx:]
y_train_val,     y_final_test     = y[:split_idx],     y[split_idx:]

print(f"Обучающая выборка: {len(X_train_val)}  тестовая: {len(X_final_test)}")

#  2. Обучение и диагностика (кросс-валидация)
print("2. Обучение и диагностика")

# Делим обучающие данные на 5 частей для кросс-валидации.
# Каждый раз одна часть — валидация, остальные 4 — обучение.
# Это даёт усреднённую и более надёжную оценку качества, чем разовый split.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Алгоритм 1: SimpleModel — наша собственная реализация.
# Простое правило: предсказывать средний рейтинг для каждого языка.
simple_model = SimpleModel()

# Алгоритм 2: RandomForestModel — обёртка над готовым сложным алгоритмом из sklearn.
# Строит 100 деревьев решений и усредняет их предсказания.
complex_model = RandomForestModel()

results_simple  = []
results_complex = []

# Цикл работает 5 раз — на каждой итерации другой блок данных идёт на валидацию
for train_idx, val_idx in kf.split(X_train_val):
    # Достаём данные и ответы для обучения и валидации
    X_t,     X_v     = X_train_val.iloc[train_idx],     X_train_val.iloc[val_idx]
    X_raw_t, X_raw_v = X_raw_train_val.iloc[train_idx], X_raw_train_val.iloc[val_idx]
    y_t,     y_v     = y_train_val.iloc[train_idx],     y_train_val.iloc[val_idx]

    # Обучаем и тестируем простую модель (получает числовые коды языков)
    simple_model.fit(X_t, y_t)
    simple_preds = simple_model.predict(X_v)
    results_simple.append(mean_absolute_error(y_v, simple_preds))

    # Обучаем и тестируем сложную модель (получает оригинальные сырые данные)
    complex_model.fit(X_raw_t, y_t)
    complex_preds = complex_model.predict(X_raw_v)
    results_complex.append(mean_absolute_error(y_v, complex_preds))

# MAE (средняя абсолютная ошибка) показывает, на сколько звёзд в среднем ошибается модель.
# Чем меньше — тем точнее. Выводим для обеих моделей, чтобы сравнить.
print(f"Средняя ошибка (MAE) простой модели:  {np.mean(results_simple):.4f}")
print(f"Средняя ошибка (MAE) сложной модели:  {np.mean(results_complex):.4f}")

# Визуализация ошибок: реальный рейтинг против предсказанного.
# Идеальная модель даёт точки вдоль красной диагональной линии.
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_v, simple_preds, alpha=0.3, color='orange', label='Simple')
plt.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'r--', lw=2)
plt.xlabel('Реальный рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Простая модель')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_v, complex_preds, alpha=0.3, color='blue', label='RFR')
plt.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'r--', lw=2)
plt.xlabel('Реальный рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Сложная модель (RFR)')
plt.legend()

plt.tight_layout()
plt.show()

# Сравнение худших предсказаний в одной таблице.
error_simple  = np.abs(y_v.values - simple_preds)
error_complex = np.abs(y_v.values - complex_preds)

comparison_df = pd.DataFrame({
    'Real':         y_v.values,
    'Simple_Pred':  simple_preds,
    'Complex_Pred': complex_preds,
    'Simple_Err':   error_simple,
    'Complex_Err':  error_complex,
})
print("\nСравнение ошибок на конкретных примерах:")
print(comparison_df.sort_values(by='Complex_Err', ascending=False).head())

#  3. Финальный отбор и сохранение
print("3. Финальный отбор и сохранение")

# Выбираем RandomForestModel, так как она лучшая по MAE.
# Обучаем её последний раз на всех доступных данных (80%), которые были для тренировки.
# Передаём X_raw — с оригинальными сырыми данными, модель масштабирует сама.
best_model = RandomForestModel()
best_model.fit(X_raw_train_val, y_train_val)

# Один раз проверяем на отложенных данных
final_preds = best_model.predict(X_raw_final_test)
final_mae   = mean_absolute_error(y_final_test, final_preds)

# Сохраняем объект класса в файл — он несёт в себе lang_map, means, stds и обученный RF.
# При загрузке через pickle.load() модель сразу готова к предсказаниям без переобучения.
with open('best_books_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Находим пример с максимальной ошибкой, чтобы показать, где модель путается чаще всего
test_errors   = np.abs(y_final_test.values - final_preds)
max_error_pos = test_errors.argmax()
X_result = y_final_test.iloc[max_error_pos]
y_result = final_preds[max_error_pos]

print(
    f"Лучшая модель — RandomForestModel. "
    f"Её ключевая метрика на новых данных — {final_mae:.4f}. "
    f"Чаще всего она путает {X_result:.1f} и {y_result:.1f}."
)