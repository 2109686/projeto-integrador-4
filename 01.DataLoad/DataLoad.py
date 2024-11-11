# Databricks notebook source
# MAGIC %run /Projeto_Integrador_IV/00.config/Settings

# COMMAND ----------

#IMPORTS
from pyspark.sql.functions import current_date, current_timestamp, expr, col, trim

# COMMAND ----------

#READ CSV CONTENT TO DATAFRAME
df = spark.read.format("csv").option("header", True).load(file_path)

for column in df.columns:
    df = df.withColumnRenamed(column, column.strip())

# Renomear a coluna "value" para "price"
df = df.withColumnRenamed("value", "price")

# COMMAND ----------

#VALIDATE DATAFRAME CONTENT
#df.display()

# COMMAND ----------

#ADD CONTROL DATE COLUMNS
df = df.withColumn("DATE_LOAD_HOUR", expr("current_timestamp() - INTERVAL 3 HOURS"))

# COMMAND ----------

#VALIDATE DATAFRAME CONTENT WITH CONTROL DATE COLUMN
#display(df)

# COMMAND ----------

# SAVE DATA DELTA FORMAT
df.write \
    .format('delta') \
    .mode('overwrite') \
    .option('mergeSchema', 'true') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(f'{database}.{table}')
print("Data written successfully!")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- VALIDATE DATA IN TABLE SQL
# MAGIC -- SELECT * FROM projeto_integrador_iv.corn_prices

# COMMAND ----------

#AVOID SMALL FILES, ORGANIZE THE FILES TO REDUCE READING PROCCESS
spark.sql(f"OPTIMIZE {database}.{table}")
print(f"Optimization process finished!")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DETAILS FROM TABLE
# MAGIC DESCRIBE DETAIL projeto_integrador_iv.corn_prices
