import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Obtém os dados do arquivo
patients_data = pd.read_csv("./src/resources/The_Cancer_data_1500_V2.csv")

# Divisão dos dados em X (features) e Y (classe)
X = patients_data.iloc[:, 1:8].values
y = patients_data.iloc[:, 8].values

# Criação do OneHotEncoder para coluna 4 (Genetic Risk)
one_hot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [4])], remainder='passthrough')
X = one_hot_encoder.fit_transform(X).toarray()

# Transformando os dados para a mesma escala
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Criação e treinamento do modelo Random Forest
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(X_train, y_train)

# Salvando modelo treinado em arquivo pkl
model_file_name = './src/resources/trained_random_forest.pkl'
pickle.dump(classifier, open(model_file_name, 'wb'))

# Salva encoders para usar na UI
scaler_encoder_file_name = './src/resources/scaler_encoder.pkl'
one_hot_encoder_file_name = './src/resources/one_hot_encoder.pkl'

pickle.dump(one_hot_encoder, open(scaler_encoder_file_name, 'wb'))
pickle.dump(scaler, open(one_hot_encoder_file_name, 'wb'))