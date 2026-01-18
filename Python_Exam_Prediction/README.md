# Предсказание прохождения экзамена

*Для проекта использовался датасет* [python_exam_perfomance](https://www.kaggle.com/datasets/emonsharkar/python-learning-and-exam-performance-dataset)

## Часть 1

Была проведена предобработка данных (данные сохранены для продолжения работы в части 2). 
Применены алгоритмы классификации:
- Логистическая регрессия `LogisticRegression`
- Метод опорных векторов `SVC` с перебором ядер в `GridSearch`
- Метод k-ближайших соседей `KNN` с перебором кол-ва соседей n

Сравнены показатели лучших моделей регрессии, SVM и KNN. На данном датасете лучший
результат показала модель SVC с линейным ядром

```
              precision    recall  f1-score   support

           0       0.90      0.97      0.94       476
           1       0.85      0.60      0.70       124

    accuracy                           0.90       600
   macro avg       0.88      0.78      0.82       600
weighted avg       0.89      0.90      0.89       600

```

## Часть 2

Для датасета были реализованы алгоритмы баггинга и бустинга.
Использованы `DecisionTreeClassifier`, `RandomForestClassifier`
и `CatBoostClassifier`. 

Лучший результат показала модель классификации catboost

```
 precision    recall  f1-score   support

           0       0.92      0.94      0.93       492
           1       0.71      0.65      0.68       108

    accuracy                           0.89       600
   macro avg       0.82      0.79      0.80       600
weighted avg       0.89      0.89      0.89       600
```

С блокнотами можно ознакомиться здесь:

[python_exam_classification](/Python_Exam_Prediction/python_exam_classification.ipynb)

[python_exam_2](/Python_Exam_Prediction/python_exam_2.ipynb)