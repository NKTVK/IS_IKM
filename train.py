import pandas as pd
import numpy as np
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
kf = KFold(n_splits=5, shuffle=True, random_state=42)
simple_model = SimpleLanguageModel() #Используем наш алгоритм
complex_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)#Исползуем RFR

results_simple = []
results_complex = []

#Кросс валидация
for train_idx, val_idx in kf.split(X_train_val):
    # Разделяем на фолды
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
# Возьмем предсказания сложной модели на последнем фолде для анализа
plt.figure(figsize=(10, 6))
plt.scatter(y_v, complex_preds, alpha=0.3, color='blue')
plt.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'r--', lw=2)
plt.xlabel('Реальный рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Диагностика ошибок: Реальные vs Предсказанные значения')
plt.show()

# Таблица примеров, где модель ошиблась сильнее всего
errors = np.abs(y_v - complex_preds)
error_analysis = pd.DataFrame({
    'Real': y_v,
    'Predicted': complex_preds,
    'Error': errors
}).sort_values(by='Error', ascending=False)