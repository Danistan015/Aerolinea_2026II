
# 03_gold_analytics.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# =========================
# Configuración
# =========================

CATALOG = "airline_mantenimiento"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

# =========================
# 1. Verificar y leer Silver
# =========================

tablas_silver = [
    "vuelos",
    "aeronaves",
    "aeropuertos",
    "mantenimientos"
]

for nombre in tablas_silver:
    tabla = f"{CATALOG}.{SILVER_SCHEMA}.{nombre}"

    if not spark.catalog.tableExists(tabla):
        raise Exception(
            f"No existe la tabla {tabla}. "
            "Ejecuta primero 02_silver_transform.py"
        )

vuelos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.vuelos")
aeronaves_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeronaves")
aeropuertos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos")
mantenimientos_slv = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos")

print("Tablas Silver verificadas correctamente")

# =========================
# 2. Crear schema Gold
# =========================

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}"
)

# =========================
# 3. Preparar aeropuertos
# =========================

origen = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("origen_id"),
    F.col("nombre").alias("aeropuerto_origen"),
    F.col("ciudad").alias("ciudad_origen"),
    F.col("pais").alias("pais_origen")
)

destino = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("destino_id"),
    F.col("nombre").alias("aeropuerto_destino"),
    F.col("ciudad").alias("ciudad_destino"),
    F.col("pais").alias("pais_destino")
)

# =========================
# 4. GOLD: Fact vuelos
# =========================

fact_vuelos = (
    vuelos_slv
    .join(aeronaves_slv, on="aeronave_id", how="left")
    .join(origen, on="origen_id", how="left")
    .join(destino, on="destino_id", how="left")
    .withColumn("anio_vuelo", F.year("fecha"))
    .withColumn("mes_vuelo", F.month("fecha"))
    .withColumn(
        "duracion_hr",
        F.round(F.col("duracion_min") / 60, 2)
    )
    .select(
        "vuelo_id",
        "fecha",
        "anio_vuelo",
        "mes_vuelo",
        "origen_id",
        "aeropuerto_origen",
        "ciudad_origen",
        "pais_origen",
        "destino_id",
        "aeropuerto_destino",
        "ciudad_destino",
        "pais_destino",
        "aeronave_id",
        "modelo",
        "fabricante",
        "anio_fabricacion",
        "estado",
        "duracion_min",
        "duracion_hr"
    )
)

# =========================
# 5. GOLD: Fact mantenimientos
# =========================

fact_mantenimientos = (
    mantenimientos_slv
    .join(aeronaves_slv, on="aeronave_id", how="left")
    .withColumn(
        "anio_mantenimiento",
        F.year("fecha")
    )
    .withColumn(
        "mes_mantenimiento",
        F.month("fecha")
    )
    .withColumn(
        "costo_por_hora_usd",
        F.when(
            F.col("duracion_hr") > 0,
            F.round(
                F.col("costo_usd") / F.col("duracion_hr"),
                2
            )
        )
    )
    .select(
        "mantenimiento_id",
        "aeronave_id",
        "modelo",
        "fabricante",
        "anio_fabricacion",
        "fecha",
        "anio_mantenimiento",
        "mes_mantenimiento",
        "tipo",
        "costo_usd",
        "duracion_hr",
        "costo_por_hora_usd"
    )
)

# =========================
# 6. GOLD: Resumen por aeronave
# =========================

resumen_mantenimiento_aeronave = (
    fact_mantenimientos
    .groupBy(
        "aeronave_id",
        "modelo",
        "fabricante",
        "anio_fabricacion"
    )
    .agg(
        F.countDistinct("mantenimiento_id")
         .alias("total_mantenimientos"),
        F.sum("costo_usd")
         .alias("costo_total_usd"),
        F.sum("duracion_hr")
         .alias("horas_totales_mantenimiento"),
        F.avg("costo_usd")
         .alias("costo_promedio_usd"),
        F.max("fecha")
         .alias("fecha_ultimo_mantenimiento")
    )
    .withColumn(
        "costo_total_usd",
        F.round("costo_total_usd", 2)
    )
    .withColumn(
        "costo_promedio_usd",
        F.round("costo_promedio_usd", 2)
    )
    .withColumn(
        "horas_totales_mantenimiento",
        F.round("horas_totales_mantenimiento", 2)
    )
)

# =========================
# 7. Ranking de costos
# =========================

ventana_costo = Window.orderBy(
    F.col("costo_total_usd").desc_nulls_last()
)

resumen_mantenimiento_aeronave = (
    resumen_mantenimiento_aeronave
    .withColumn(
        "ranking_costo",
        F.dense_rank().over(ventana_costo)
    )
)

# =========================
# 8. Guardar tablas Gold
# =========================

tablas_gold = {
    "fact_vuelos": fact_vuelos,
    "fact_mantenimientos": fact_mantenimientos,
    "resumen_mantenimiento_aeronave": resumen_mantenimiento_aeronave
}

for nombre, df in tablas_gold.items():
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.{nombre}")
    )

    print(f"Tabla Gold guardada: {nombre}")

# =========================
# 9. Validar resultados
# =========================

for nombre in tablas_gold:
    tabla = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.{nombre}")

    total = tabla.count()

    print(f"\nTabla Gold: {nombre}")
    print(f"Total registros: {total}")

    if total == 0:
        raise ValueError(
            f"La tabla {nombre} no contiene registros"
        )

    tabla.show(5, truncate=False)

# =========================
# 10. Validaciones adicionales
# =========================

# Revisar vuelos con origen y destino iguales

vuelos_mismo_aeropuerto = fact_vuelos.filter(
    F.col("origen_id") == F.col("destino_id")
)

print(
    "Vuelos con origen y destino iguales:",
    vuelos_mismo_aeropuerto.count()
)

vuelos_mismo_aeropuerto.select(
    "vuelo_id",
    "fecha",
    "origen_id",
    "destino_id",
    "estado"
).orderBy("vuelo_id").show(100, truncate=False)

# Revisar identificadores nulos

nulos_vuelos = fact_vuelos.filter(
    F.col("vuelo_id").isNull()
).count()

nulos_mantenimientos = fact_mantenimientos.filter(
    F.col("mantenimiento_id").isNull()
).count()

print("Identificadores nulos en vuelos:", nulos_vuelos)
print(
    "Identificadores nulos en mantenimientos:",
    nulos_mantenimientos
)

if nulos_vuelos > 0 or nulos_mantenimientos > 0:
    raise ValueError(
        "Se encontraron identificadores nulos en las tablas Gold"
    )

print("Proceso Gold finalizado correctamente")
