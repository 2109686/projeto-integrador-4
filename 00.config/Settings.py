# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS spark_catalog.projeto_integrador_iv
# MAGIC LOCATION 'dbfs/FileStore/Projeto_Integrador_IV/projeto_integrador_iv/';

# COMMAND ----------

#VARIABLES DEFINITIONS
database = 'projeto_integrador_iv'
table = 'corn_prices'
file_path = 'dbfs:/FileStore/Projeto_Integrador_IV/corn_prices_historical_chart_data-1.csv'

# COMMAND ----------


