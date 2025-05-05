'''
Titolo: Previsione delle calorie bruciate in una sessione di corsa

Descrizione: Partendo dal documento condiviso:

1.	Pulire e preparare i dati:
    - Analisi esplorativa (Dettagliare l'esito dell'analisi)
    - estrarre le feature principali (durata, distanza, velocità media, elevazione guadagnata, battito medio, etc.);
    - gestire eventuali dati mancanti.

2.	Costruire un modello supervised (con scikit-learn):
    - regressione per prevedere le calorie bruciate sulla base delle altre variabili.
    - divisione train/test e valutazione con MSE o R².

3.	In ambito Unsupervised quali algoritmi si potrebbero applicare e con quale fine. Argomentare la risposta.

4.	Organizzare il progetto:
    - seguendo principi di software engineering (struttura a moduli, funzioni ben definite);
'''


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

raw_data_activities = pd.read_json('activities_2024-12-13_to_2025-04-16 1.json')

clean_data_activities = raw_data_activities[['duration', 'distance', 'averageSpeed', 'elevationGain', 'averageHR', 'calories']]



'''
# print(raw_data_activities.describe(include='all'))

nan_values_per_feature = raw_data_activities.isna().sum()
percentage = 25
missing_values_percentage = raw_data_activities.shape[0] * (percentage / 100)

column_names_to_drop = [column for column in nan_values_per_feature.index if nan_values_per_feature[column] > missing_values_percentage]
raw_data_activities = raw_data_activities.drop(columns=column_names_to_drop)

nan_values_per_sample = raw_data_activities.isna().sum(axis=1)
percentage = 25
missing_values_percentage = raw_data_activities.shape[1] * (percentage / 100)

rows_to_drop = [row for row in nan_values_per_sample.index if nan_values_per_sample[row] > missing_values_percentage]
raw_data_activities = raw_data_activities.drop(rows_to_drop)
'''
object_variables = [column for column in raw_data_activities.columns
                    if not is_numeric_dtype(raw_data_activities[column].dtype)
                    or is_bool_dtype(raw_data_activities[column].dtype)]

raw_data_activities = raw_data_activities.drop(columns=object_variables)
'''
# durata, distanza, velocità media, elevazione guadagnata, battito medio ecc.
'''

imputer = SimpleImputer(strategy='mean', copy=False)
clean_data_activities = pd.DataFrame(imputer.fit_transform(clean_data_activities), columns=clean_data_activities.columns)
print(clean_data_activities)

X = clean_data_activities.drop(columns='calories')
y = clean_data_activities['calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

scaler =  StandardScaler()
Xtrain = scaler.fit_transform(X_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)

plt.scatter(y_test, y_pred)
plt.grid(visible=True, alpha=0.2)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()




