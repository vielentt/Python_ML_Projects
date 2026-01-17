# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

"""Работа с данными агрегатора такси. В зависимости от характеристик поездки требуется предсказать один из трех типов повышенного ценообразования: [1, 2, 3]. Таким образом, это поможет компании оптимально мэтчить такси и клиентов."""

df = pd.read_csv('sigma_cabs.csv')
df.shape

df = df.set_index('Trip_ID')
df.head()

"""Описание признаков:

1. **Trip_ID**: ID поездки
2. **Trip_Distance**: Расстояние опездки по первоначальному запросу клиента
3. **TypeofCab**: Категория такси, запрошенная клиентом
4. **CustomerSinceMonths**: С какого месяца клиент использует сервис. 0 - с текущего
5. **LifeStyleIndex**: Внутренний индекс сервиса стиля жизни клиента, исходя из внутренних данных
6. **ConfidenceLifeStyle_Index**: Категория отражающая уверенность   индекса LifeStyleIndex
7. **Destination_Type**: Один из 14 типов финальной точки. Внутренняя сегментация сервиса
8. **Customer_Rating**: Средний рейтинг сервиса клиентом
9. **CancellationLast1Month**: Количество отмененных клиентом поездок за последний месяц
10. **Var1**, **Var2** and **Var3**: Скрытые категориальные переменные, которые могут быть использованы для моделирования
11. **Gender**: Пол клиента

**SurgePricingType**: Целевая переменная - 3 класса

### EDA
"""

# заполнение NA (медианным значением для числовых)
df.isna().sum()

df['Customer_Since_Months'].describe()

df['Customer_Since_Months'] = df['Customer_Since_Months'].fillna(df['Customer_Since_Months'].median())

df['Life_Style_Index'].describe()

sns.boxplot(df['Life_Style_Index'])

df['Life_Style_Index'].fillna(df['Life_Style_Index'].median()).describe()

df['Life_Style_Index'] = df['Life_Style_Index'].fillna(df['Life_Style_Index'].median())

df['Var1'].describe()

sns.boxplot(df['Var1'])

sum(df['Var1']>175)

df = df[(df['Var1'] < 175) | df['Var1'].isna()]

df.shape

df['Var1'].describe()

df['Var1'].fillna(df['Var1'].median()).describe()
# здесь std сильно меняется, принято решение замены NA средним значением

df['Var1'] = df['Var1'].fillna(df['Var1'].mean())

df.isna().sum()

# замена каегориальных пропусков наиболее популярным
df['Type_of_Cab'].value_counts()

df['Type_of_Cab'].fillna('B').value_counts()

df['Confidence_Life_Style_Index'].value_counts()

df['Confidence_Life_Style_Index'].fillna('B').value_counts()

df['Confidence_Life_Style_Index'] = df['Confidence_Life_Style_Index'].fillna('B')
df['Type_of_Cab'] = df['Type_of_Cab'].fillna('B')

df.isna().sum()

df.describe()

df.info()
# проверка типов данных

# эти данные не могут быть не целыми
df['Customer_Since_Months'] = df['Customer_Since_Months'].astype('int16')
df['Cancellation_Last_1Month'] = df['Cancellation_Last_1Month'].astype('int16')

numerical = df.describe().columns
numerical = numerical.delete(-1)
numerical

categorical = df.describe(include='object').columns
categorical

# матрица корреляций числовых переменных
sns.heatmap(df.corr(numeric_only=True), annot=True)

# топ5 пар самых коррелированных признаков
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

get_top_abs_correlations(df[numerical], 5)
# сильной корреляции нет

# Квазиконстантность
from sklearn.feature_selection import VarianceThreshold
variance = VarianceThreshold(threshold=1)
variance.fit_transform(df[numerical])
variance.get_feature_names_out()

numerical
# 'Life_Style_Inde' и 'Customer_Rating' - признаки с низкой дисперсией. Пока будем считать, что все признаки важны для модели.

# ящики с усами для числовых признаков
for col in numerical:
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    sns.boxplot(y=col, x=df['Surge_Pricing_Type'].astype('category'), data=df)
    plt.show()
# 'Customer_Since_Months' нет различий, принято решение удалить признак
# Var2 небольшие различия

df = df.drop(['Customer_Since_Months'], axis = 1)

numerical = numerical.drop('Customer_Since_Months')

df.head()

# ищем разнообразия для категориальных переменных
for col in categorical:
    g = sns.catplot(x=col, kind='count', col=df['Surge_Pricing_Type'], data=df)
    g.set_xticklabels(rotation=60)
# все переменные различаются для разных признаков

# выделяем признаки и таргет
X = df.drop('Surge_Pricing_Type', axis=1)
Y = df['Surge_Pricing_Type']

X

from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# OneHotEncoder и TargetEncoder для категориальных переменных
# (в зависимости от кол-ва уникальных значений)

ohe = []
mte = []

for col in categorical:
  if df[col].nunique() <= 4:
    ohe.append(col)
  else:
    mte.append(col)

categorical

mte

transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False, drop='first'), ohe),
    ('mte', TargetEncoder(), mte)],
    remainder='passthrough')

transformer.set_output(transform="pandas")

X_transformed = transformer.fit_transform(X, Y)

X_transformed
# проверка: 7 цифровых, 2 ohe 2 mte

"""### Training"""

np.random.seed(2022)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test  = train_test_split(X_transformed, Y,
                                                     test_size=0.2,
                                                     shuffle=True,
                                                     random_state=2022)

"""Обучение One-vs-Rest Logreg, подсчет метрик"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([("scaler", StandardScaler()),
                 ("one_vs_rest", OneVsRestClassifier(LogisticRegression(class_weight='balanced')))])
pipe.fit(X_train, Y_train)

from sklearn.metrics import classification_report

OneVSRestLogreg1 = classification_report(Y_test, pipe.predict(X_test), digits=3)
print(classification_report(Y_test, pipe.predict(X_test), digits=3))

"""Подбор оптимальных гиперпараметров модели с помощью `GridSearchCV()`

"""

param_grid = {'one_vs_all__estimator__penalty': ['l2', 'elasticnet'],
              'one_vs_all__estimator__C': [0.001, 0.01, 0.1, 1]}

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

pipe = Pipeline([("scaler", StandardScaler()),
                 ("one_vs_all", OneVsRestClassifier(LogisticRegression(class_weight='balanced')))])
grid = GridSearchCV(pipe, param_grid, scoring=accuracy_score)
grid.fit(X_train, Y_train)

print(grid.best_params_)

OneVSRestGrid2 = classification_report(Y_test, grid.predict(X_test), digits=3)
print(classification_report(Y_test, grid.predict(X_test), digits=3))

"""Калибровочные кривые для Logistic Classifier: 0-vs-rest, 1-vs-rest, 2-vs-rest

"""

proba = grid.predict_proba(X_test)

from sklearn.calibration import CalibrationDisplay

for i in [1, 2, 3]:
  Y_true = (Y_test == i)
  Y_prob = proba[:, i-1]
  CalibrationDisplay.from_predictions(Y_true, Y_prob, name=f'{i}-vs-rest')

"""One-vs-One `SGDClassifier`"""

X_train, X_test, y_train, y_test  = train_test_split(X_transformed, Y,
                                                     test_size=0.2,
                                                     shuffle=True,
                                                     random_state=2022)

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier

pipe_ovo = Pipeline([("scaler", StandardScaler()),
                 ("one_vs_one", OneVsOneClassifier(SGDClassifier()))])

pipe_ovo.fit(X_train, y_train)

OneVSOne4 = classification_report(y_test, pipe_ovo.predict(X_test), digits=3)
print(classification_report(y_test, pipe_ovo.predict(X_test), digits=3))

"""Подбор оптимальных гиперпараметров модели с помощью `GridSearchCV()` с перебором функций потерь."""

param_grid = {'one_vs_one__estimator__loss': ['hinge', 'log', 'modified_huber'],
              'one_vs_one__estimator__penalty': ['l1', 'l2'],
              'one_vs_one__estimator__alpha': [0.001, 0.01, 0.1]}

grid = GridSearchCV(pipe_ovo, param_grid, scoring=accuracy_score)
grid.fit(X_train, y_train)

print(grid.best_params_)

OneVSOneGrid5 = classification_report(Y_test, grid.predict(X_test), digits=3)
print(classification_report(Y_test, grid.predict(X_test), digits=3))