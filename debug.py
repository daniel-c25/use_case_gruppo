import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datHandler.datHandler as dh # TODO: Documentation needed for package usage

# Create a DataFrame
garmin_readings = pd.read_json('activities_2024-12-13_to_2025-04-16 1.json')

# TODO: EDA and analysis summary needed

# Important features extraction: durata, distanza, velocità media, elevazione guadagnata, battito medio, calorie

features = ['duration', 'distance', 'averageSpeed', 'elevationGain', 'averageHR', 'calories']
important_features_df = pd.DataFrame(dh.features_extractor(garmin_readings, features), columns=features)
print(important_features_df.head)

# Missing data handling
df_missing_data = ((important_features_df.isnull().sum(axis=0))
                   .rename_axis('features')
                   .reset_index(name='missing values'))

features = df_missing_data['features']
missing_values = df_missing_data['missing values']

# Creating bar chart
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
bars = ax.bar(features, missing_values, color="skyblue")

# Adding labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, str(yval), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customizing the plot
ax.set_title("Features and their missing Values")
plt.yticks([])
plt.xticks(rotation=45)  # Rotate labels for readability

plt.show()

# TODO: eplain the next operation of missing data imputation on the data analisis

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
clean_data = pd.DataFrame(imputer.fit_transform(important_features_df))
clean_data.columns = df_missing_data['features'].values
if clean_data.isna().sum().sum() == 0: #check of features free of missing or categorical data
    print("No missing values")
else:
    print("There are {} missing values.".format(clean_data.isna().sum().sum()))


# data splitting in train and test

X = clean_data.drop('calories', axis=1)
y = clean_data['calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72)

# scaling before fitting in the linear regression model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fitting into a linear regression model

linear_regression = LinearRegression()
linear_regression.fit(X_train_scaled, y_train)
y_pred = linear_regression.predict(X_test_scaled)

# plotting the result of the predictions

plt.clf()
plt.scatter(y_pred, y_test, color='red')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', alpha=0.5, linestyle='dashed')
plt.xlabel('Predicted burned calories')
plt.ylabel('Actual burned calories')
plt.grid(visible=True, alpha=0.2)
plt.show()

# root mean squared error and R^2 calculation together with standard deviation

std = np.std(y_test)
rmse = mean_squared_error(y_test, y_pred)
score = linear_regression.score(X_test_scaled, y_test)
print("R^2 value: {}".format(score),
      "\nMean squared error value: {}".format(rmse),
      '\nStandard deviation value: {}'.format(std))

# TODO: at this point explain every step of the supervised learning processes
# TODO: file config con i parametri standard delle funzioni

# Unsupervised learning algorithms: KMeans for clustering...
kmeans = KMeans(n_clusters=2, random_state=7)
kmeans.fit(X_train)
predictions = kmeans.predict(X_test)
centers = kmeans.cluster_centers_

# PCA for dimension reduction...
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train.values)
X_test_pca = pca.transform(X_test.values)
centers_pca = pca.transform(centers)

# t-SNE for relational awarness
tsne = TSNE(n_components=2, random_state=7)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=predictions, alpha=.5)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='d', color='cyan')
plt.show()

plt.clf()
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=kmeans.labels_, alpha=0.5)
plt.show()
