# Databricks notebook source
#IMPORTS
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# COMMAND ----------

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
