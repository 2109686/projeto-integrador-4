# Databricks notebook source
#IMPORTS
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

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
