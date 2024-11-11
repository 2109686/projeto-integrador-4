# Databricks notebook source
# MAGIC %pip install statsmodels pmdarima

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

#IMPORTS
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# COMMAND ----------

# MAGIC %run /Projeto_Integrador_IV/00.config/Settings

# COMMAND ----------

# Carregar os dados da tabela Delta
df = spark.table(f"{database}.{table}")

# Converter os dados Spark para Pandas
df_pd = df.toPandas()
df_pd.set_index('date', inplace=True)

# Garantir que 'price' seja numérica
df_pd['price'] = pd.to_numeric(df_pd['price'], errors='coerce')

df_pd.index = pd.to_datetime(df_pd.index)
df_pd = df_pd.asfreq('D')  # Se os dados são diários, defina 'D' como a frequência

# Treinar o modelo ARIMA
arima_model = ARIMA(df_pd['price'], order=(5,1,0))  # Ajuste os parâmetros (p,d,q) conforme necessário
arima_fit = arima_model.fit()

# Previsão
arima_forecast = arima_fit.forecast(steps=30)  # Exemplo: previsão para os próximos 30 dias
print(arima_forecast)
