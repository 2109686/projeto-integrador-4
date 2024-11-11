# Databricks notebook source
# MAGIC %pip install statsmodels pmdarima

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

from statsmodels.tsa.arima.model import ARIMA

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# COMMAND ----------

# MAGIC %run /Projeto_Integrador_IV/01.DataLoad/DataLoad

# COMMAND ----------

# MAGIC %run /Projeto_Integrador_IV/02.DataPreparation/DataPreparation

# COMMAND ----------

# Regressão Linear

# Carregar os dados de treino e teste do arquivo Parquet
train_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/train_data.parquet")
test_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/test_data.parquet")

# Converter a coluna "price" para o tipo numérico (DoubleType)
train_data = train_data.withColumn("price", col("price").cast(DoubleType()))
test_data = test_data.withColumn("price", col("price").cast(DoubleType()))

# Remover linhas com valores nulos ou NaN na coluna "price"
train_data = train_data.dropna(subset=["price"])
test_data = test_data.dropna(subset=["price"])

# Modelo de Regressão Linear
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(train_data)

# Avaliação do modelo
lr_predictions = lr_model.transform(test_data)
lr_evaluator = lr_model.summary

print("Erro Médio Absoluto (MAE):", lr_evaluator.meanAbsoluteError)
print("Erro Quadrático Médio (MSE):", lr_evaluator.meanSquaredError)

# COMMAND ----------

# ARIMA

# Carregar os dados da tabela Delta
df = spark.table(f"{database}.{table}")

# Converter os dados Spark para Pandas
df_pd = df.toPandas()
df_pd.set_index('date', inplace=True)

# Garantir que 'price' seja numérica
df_pd['price'] = pd.to_numeric(df_pd['price'], errors='coerce')

df_pd.index = pd.to_datetime(df_pd.index)
df_pd = df_pd.asfreq('D')  # Se os dados são diários, defina 'D' como a frequência

# Remover valores NaN dos dados antes de treinar o modelo
df_pd = df_pd.dropna()

# Treinar o modelo ARIMA
arima_model = ARIMA(df_pd['price'], order=(5,1,0))  # Ajuste os parâmetros (p,d,q) conforme necessário
arima_fit = arima_model.fit()

# Previsão
arima_forecast = arima_fit.forecast(steps=30)  # Exemplo: previsão para os próximos 30 dias
#print(arima_forecast)

# COMMAND ----------


# RANDOM FOREST


# Carregar os dados de treino e teste do arquivo Parquet
train_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/train_data.parquet")
test_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/test_data.parquet")

# Converter a coluna "price" para o tipo numérico (DoubleType)
train_data = train_data.withColumn("price", col("price").cast(DoubleType()))
test_data = test_data.withColumn("price", col("price").cast(DoubleType()))

# Modelo de Floresta Aleatória
rf = RandomForestRegressor(featuresCol="features", labelCol="price", numTrees=100)
rf_model = rf.fit(train_data)

# Avaliação do modelo
rf_predictions = rf_model.transform(test_data)

# Criação do avaliador
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

# Calcular o RMSE (Root Mean Squared Error)
rmse = evaluator.evaluate(rf_predictions)
print("RMSE:", rmse)

# Você pode calcular outras métricas, como R2, MAE, etc.
r2_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(rf_predictions)
print("R2:", r2)

# Aqui você pode calcular métricas de avaliação manualmente, se necessário
print("Importância das Features:", rf_model.featureImportances)

# COMMAND ----------

# SVR

# 1. Carregar os dados de treino e teste diretamente em Spark DataFrames
train_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/train_data.parquet")
test_data = spark.read.parquet("/FileStore/Projeto_Integrador_IV/test_data.parquet")

# 2. Converter Spark DataFrames para arrays NumPy para compatibilidade com scikit-learn
# Supondo que `train_data` e `test_data` têm uma coluna "price" e uma coluna "date" transformada para "features" com dias sequenciais

# Extrair 'features' e 'price' como arrays NumPy para treino e teste
X_train = np.array(train_data.select("features").rdd.flatMap(lambda x: x).collect()).reshape(-1, 1)
y_train = np.array(train_data.select("price").rdd.flatMap(lambda x: x).collect())

X_test = np.array(test_data.select("features").rdd.flatMap(lambda x: x).collect()).reshape(-1, 1)
y_test = np.array(test_data.select("price").rdd.flatMap(lambda x: x).collect())

# 3. Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 4. Criar e treinar o modelo SVR
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_train, y_train)

# 5. Fazer previsões no conjunto de teste
y_pred = svr.predict(X_test)

# Reverter a normalização dos valores de y para comparar com os valores reais
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 6. Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Exibir as métricas de avaliação
print(f"MAE (Erro Absoluto Médio): {mae}")
print(f"MSE (Erro Quadrático Médio): {mse}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse}")

# 7. Visualizar as previsões
#plt.figure(figsize=(10, 6))
#plt.scatter(X_test, y_test, color='red', label='Dados Reais')
#plt.scatter(X_test, y_pred, color='blue', label='Previsões')
#plt.title('SVR - Comparação de Previsões e Dados Reais')
#plt.xlabel('Data (representada como feature)')
#plt.ylabel('Preço do Milho')
#plt.legend()
#plt.show()

# COMMAND ----------

#Comparação

# Dicionário para armazenar as métricas de cada modelo
resultados_modelos = {}

# Avaliação do modelo de Regressão Linear
mae_lr = lr_evaluator.meanAbsoluteError
mse_lr = lr_evaluator.meanSquaredError
rmse_lr = np.sqrt(mse_lr)
resultados_modelos['Regressão Linear'] = {'MAE': mae_lr, 'MSE': mse_lr, 'RMSE': rmse_lr}

# Avaliação do modelo ARIMA
# Para ARIMA, usamos previsões diretas em `arima_forecast` com base no histórico `df_pd['price']`
y_test_arima = df_pd['price'].iloc[-30:]  # Últimos 30 dias de dados reais, supondo previsão de 30 passos
mae_arima = mean_absolute_error(y_test_arima, arima_forecast)
mse_arima = mean_squared_error(y_test_arima, arima_forecast)
rmse_arima = np.sqrt(mse_arima)
resultados_modelos['ARIMA'] = {'MAE': mae_arima, 'MSE': mse_arima, 'RMSE': rmse_arima}

# Avaliação do modelo Random Forest
rmse_rf = rmse  # Já calculado anteriormente no código do Random Forest
r2_rf = r2      # R2 do Random Forest
# Cálculo manual para MAE e MSE usando as previsões do modelo
mae_rf = mean_absolute_error(test_data.select("price").toPandas(), rf_predictions.select("prediction").toPandas())
mse_rf = mean_squared_error(test_data.select("price").toPandas(), rf_predictions.select("prediction").toPandas())
resultados_modelos['Random Forest'] = {'MAE': mae_rf, 'MSE': mse_rf, 'RMSE': rmse_rf, 'R2': r2_rf}

# Avaliação do modelo SVR
mae_svr = mean_absolute_error(y_test, y_pred)
mse_svr = mean_squared_error(y_test, y_pred)
rmse_svr = np.sqrt(mse_svr)
resultados_modelos['SVR'] = {'MAE': mae_svr, 'MSE': mse_svr, 'RMSE': rmse_svr}

# Criar um DataFrame com os resultados para comparação
df_resultados = pd.DataFrame(resultados_modelos).T
print(df_resultados)

# Visualização das métricas de erro para cada modelo
plt.figure(figsize=(12, 6))
df_resultados[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 5), title='Comparação de Modelos - MAE e RMSE')
plt.ylabel('Erro')
plt.xlabel('Modelos')
plt.xticks(rotation=0)
plt.show()

# Comparação das previsões (Gráfico de dispersão)
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='red', label='Dados Reais')
plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Previsões SVR')
plt.plot(range(len(y_test_arima)), y_test_arima, color='green', label='Previsões ARIMA')
plt.plot(range(len(y_test_arima)), arima_forecast, color='purple', linestyle='--', label='Previsões ARIMA')
#plt.scatter(test_data.select("features").toPandas(), rf_predictions.select("prediction").toPandas(), color='orange', label='Previsões Random Forest')
plt.scatter(range(len(lr_predictions.select("prediction").toPandas())), lr_predictions.select("prediction").toPandas(), color='cyan', label='Previsões Regressão Linear')
plt.title('Comparação de Previsões - Dados Reais vs Modelos')
plt.xlabel('Observações')
plt.ylabel('Preço')
plt.legend()
plt.show()
