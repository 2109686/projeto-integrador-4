# Databricks notebook source
# MAGIC %run /Projeto_Integrador_IV/00.config/Settings

# COMMAND ----------

#IMPORTS
from pyspark.sql.functions import to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# COMMAND ----------

# Carregar os dados da tabela Delta
df = spark.table(f"{database}.{table}")

# Converter a coluna "date" para o tipo Date, se necessário
df = df.withColumn("date", to_date("date", "yyyy-MM-dd"))

# Ordenar os dados pela data
df = df.orderBy("date")

df = df.withColumn("index", row_number().over(Window.orderBy("date")))

# Preparar os dados para o modelo
assembler = VectorAssembler(inputCols=["index"], outputCol="features")
data = assembler.transform(df).select("features", "price")

# Dividir os dados em treino e teste
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Salvar os dados de treino e teste como arquivos Parquet
train_data.write.mode("overwrite").parquet("/FileStore/Projeto_Integrador_IV/train_data.parquet")
test_data.write.mode("overwrite").parquet("/FileStore/Projeto_Integrador_IV/test_data.parquet")
