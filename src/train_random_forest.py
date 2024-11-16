import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


# Obtém os dados do arquivo
patients_data = pd.read_csv("./src/resources/The_Cancer_data_1500_V2.csv")

# Divisão dos dados em X (features) e Y (classe)
X = patients_data.iloc[:, 0:8].values
y = patients_data.iloc[:, 8].values

# Criação do OneHotEncoder para coluna 4 (Genetic Risk)
one_hot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [4])], remainder='passthrough')
X = one_hot_encoder.fit_transform(X)

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

pickle.dump(one_hot_encoder, open(one_hot_encoder_file_name, 'wb'))
pickle.dump(scaler, open(scaler_encoder_file_name, 'wb'))

# Verificação de Métricas.

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {accuracy * 100}%")
print(f"Precisão: {precision * 100}%")
print(f"F1-Score: {f1 * 100}%")
print(f"Recall: {recall * 100}%")