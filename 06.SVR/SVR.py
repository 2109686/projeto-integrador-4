# Databricks notebook source
#IMPORTS
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# COMMAND ----------


# Carregar os dados de treino e teste do arquivo Parquet
train_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/train_data.parquet")
test_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/test_data.parquet")

# Exemplo de como preparar seus dados no formato esperado pelo scikit-learn
# Aqui, 'train_data' e 'test_data' são DataFrames do Spark, então vamos primeiro
# converter as colunas para arrays ou DataFrames do pandas

# Supondo que 'train_data' tenha as colunas 'features' e 'price'
# Convertendo o 'features' para um formato utilizável pelo scikit-learn
X_train = np.array(train_data.select("features").rdd.map(lambda row: row[0]).collect())  # Converte a coluna 'features'
y_train = np.array(train_data.select("price").rdd.map(lambda row: row[0]).collect())  # Converte a coluna 'price'

X_test = np.array(test_data.select("features").rdd.map(lambda row: row[0]).collect())  # Converte a coluna 'features' para o conjunto de teste
y_test = np.array(test_data.select("price").rdd.map(lambda row: row[0]).collect())  # Converte a coluna 'price' para o conjunto de teste

# Normalizando os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Criar o modelo SVR
svr = SVR(kernel='rbf', C=100, epsilon=0.1)

# Treinar o modelo
svr.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = svr.predict(X_test)

# Reverter a normalização dos valores de y
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Exibir as métricas de avaliação
print(f"MAE (Erro Absoluto Médio): {mae}")
print(f"MSE (Erro Quadrático Médio): {mse}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse}")

# Visualizar as previsões
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='red', label='Dados Reais')
plt.scatter(X_test, y_pred, color='blue', label='Previsões')
plt.title('SVR - Comparação de Previsões e Dados Reais')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
