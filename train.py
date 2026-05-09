import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#Класс для нашего алгоритма (прогноз - средений рейтинг для каждого языка)
class SimpleLanguageModel:
    def __init__(self):
        self.lang_means = {} #язык:средний рейтинг
        self.global_mean = 0 #средний рейтинг по всем книгам на всякий случай

    def fit(self, X_train, y_train):
        data = pd.DataFrame({'lang': X_train['language_code'], 'rating': y_train}) #язык и ответы
        self.lang_means = data.groupby('lang')['rating'].mean().to_dict() #к каждому языку его средний рейтинг считаем
        self.global_mean = y_train.mean() #Среднее по всей выборке на всякий случай

    def predict(self, X_test):
        #Для каждого языка берем его среднее, если языка нет, то общее среднее
        return X_test['language_code'].map(self.lang_means).fillna(self.global_mean).values

#1. Подготовка данных
print("---------------------1. Подготовка данных-----------------------")

#Загрузка файла + пропуск грязных данных
df = pd.read_csv('books.csv', on_bad_lines="skip")
#Очистка от лишних пробелов
df.columns = [col.strip() for col in df.columns]

#Признаки(X): num_pages (кол-во страниц), ratings_count(кол-во оценок), language_code(язык), text_reviews_count(кол-во отзывов)
#Правильный ответ(y): average_rating(средняя оценка)

#Выбираем нужные нам столбцы
features = ['num_pages', 'ratings_count', 'text_reviews_count', 'language_code']
target = 'average_rating'
X = df[features].copy()
y = df[target].copy()

#Превращаем категориальные признаки в числовые
unique_languages = X['language_code'].unique()
lang_map = {lang: i for i, lang in enumerate(unique_languages)}
X['language_code'] = X['language_code'].map(lang_map)

#Масштабирование числовых признаков по формуле ((x - mean) / std) вычитаем среднее и делим на стандартное отклонение
numeric_cols = ['num_pages', 'ratings_count', 'text_reviews_count']
for col in numeric_cols:
    mean_val = X[col].mean()
    std_val = X[col].std()
    X[col] = (X[col] - mean_val) / std_val

#Разделение данных
indices = np.arange(len(X))
np.random.seed(42) #Для воспроизводимости
np.random.shuffle(indices)

X = X.iloc[indices]
y = y.iloc[indices]

#Разделяем 80% на обучающую и 20% на тестовую. 80 на 20 хорошее разбиение для среднего датасета, как у нас.
#Если брать 75/25, то разница для теста ничего не даст, а вот для обучения потеря может быть немного значительной
split_idx = int(len(X) * 0.8)

X_train_val, X_final_test = X[:split_idx], X[split_idx:]
y_train_val, y_final_test = y[:split_idx], y[split_idx:]

print(f"Обучающая выборка: {len(X_train_val)} тестовая: {len(X_final_test)}")

#2.Обучение и диагностика
print("---------------------2.Обучение и диагностика-----------------------")
kf = KFold(n_splits=5, shuffle=True, random_state=42)#делим данные на 5 частей
simple_model = SimpleLanguageModel() #Используем наш алгоритм
complex_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)#Исползуем RFR

results_simple = []
results_complex = []

#Кросс валидация
for train_idx, val_idx in kf.split(X_train_val): #Цикл рабоатет 5 раз, проходимся по всем данным
    # Достаём даннеы и ответы для обучения
    X_t, X_v = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_t, y_v = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    # Обучаем и тестируем простую модель
    simple_model.fit(X_t, y_t)
    simple_preds = simple_model.predict(X_v)
    results_simple.append(mean_absolute_error(y_v, simple_preds))

    # Обучаем и тестируем сложную модель
    complex_model.fit(X_t, y_t)
    complex_preds = complex_model.predict(X_v)
    results_complex.append(mean_absolute_error(y_v, complex_preds))

print(f"Средняя ошибка (MAE) простой модели: {np.mean(results_simple):.4f}")
print(f"Средняя ошибка (MAE) сложной модели: {np.mean(results_complex):.4f}")

#Визаулизация ошибок
plt.figure(figsize=(12, 6))

# Левый график: Простая модель
plt.subplot(1, 2, 1)
plt.scatter(y_v, simple_preds, alpha=0.3, color='orange', label='Simple')
plt.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'r--', lw=2)
plt.xlabel('Реальный рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Ошибки')
plt.legend()

# Правый график: Сложная модель (RFR)
plt.subplot(1, 2, 2)
plt.scatter(y_v, complex_preds, alpha=0.3, color='blue', label='RFR')
plt.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'r--', lw=2)
plt.xlabel('Реальный рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Ошибки')
plt.legend()

plt.tight_layout()
plt.show()

# --- Сравнение худших предсказаний в одной таблице ---
error_simple = np.abs(y_v - simple_preds)
error_complex = np.abs(y_v - complex_preds)

comparison_df = pd.DataFrame({
    'Real': y_v,
    'Simple_Pred': simple_preds,
    'Complex_Pred': complex_preds,
    'Simple_Err': error_simple,
    'Complex_Err': error_complex
})

# Сортируем по ошибке сложной модели, чтобы увидеть, где она все еще пасует
print("\nСравнение ошибок на конкретных примерах:")
print(comparison_df.sort_values(by='Complex_Err', ascending=False).head())

#3.Финальный отбор и сохранение
print("---------------------3.Финальный отбор и сохранение-----------------------")
# Выбираем RFR, так как она лучшая по MAE
best_model = complex_model

# Обучаем её последний раз на всех доступных данных (80%), которые были для тренировки
best_model.fit(X_train_val, y_train_val)

# Проверяем на отложенных данных ("сейф")
final_preds = best_model.predict(X_final_test)
final_mae = mean_absolute_error(y_final_test, final_preds)

# Сохранение модели в файл
with open('best_books_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
# Считаем разницу между фактом и прогнозом для каждого примера в тесте
test_errors = np.abs(y_final_test - final_preds)

# Находим индекс строки, где ошибка максимальна
max_error_idx = test_errors.idxmax()

# Достаем реальное значение (X) и что нагадала модель (Y) для этой строки
X_result = y_final_test.loc[max_error_idx]
y_result = final_preds[np.where(y_final_test.index == max_error_idx)[0][0]]

# Финал
print(f"Лучшая модель — RandomForestRegressor. Её ключевая метрика на новых данных — {final_mae:.4f}. Чаще всего она путает {X_result:.1f} и {y_result:.1f}.")