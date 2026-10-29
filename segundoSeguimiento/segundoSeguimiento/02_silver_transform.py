
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# =========================
# Configuración
# =========================

CATALOG = "airline_mantenimiento"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"

spark = SparkSession.builder.getOrCreate()

# =========================
# 0. Asegurar schema Silver
# =========================

spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}"
)

print("Schema Silver preparado correctamente")

# =========================
# 1. Verificar y leer Bronze
# =========================

tablas_bronze = [
    "vuelos",
    "aeronaves",
    "aeropuertos",
    "mantenimientos"
]

for nombre in tablas_bronze:
    tabla = f"{CATALOG}.{BRONZE_SCHEMA}.{nombre}"

    if not spark.catalog.tableExists(tabla):
        raise Exception(
            f"No existe la tabla {tabla}. "
            "Ejecuta primero 01_bronze_ingestion.py"
        )

print("Tablas Bronze verificadas correctamente")

vuelos_bz = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.vuelos"
)

aeronaves_bz = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.aeronaves"
)

aeropuertos_bz = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.aeropuertos"
)

mantenimientos_bz = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.mantenimientos"
)

# =========================
# 2. Transformaciones Silver
# =========================

# Transformaciones Silver vuelos

vuelos_slv = (
    vuelos_bz
    .select(
        "vuelo_id",
        "fecha",
        "origen_id",
        "destino_id",
        "aeronave_id",
        "estado",
        "duracion_min"
    )
    .withColumn("fecha", F.to_date(F.col("fecha")))
    .withColumn("duracion_min", F.col("duracion_min").cast("int"))
    .dropDuplicates(["vuelo_id"])
)

# Transformaciones Silver aeronaves

aeronaves_slv = (
    aeronaves_bz
    .select(
        "aeronave_id",
        "modelo",
        "fabricante",
        "anio_fabricacion"
    )
    .withColumn(
        "anio_fabricacion",
        F.col("anio_fabricacion").cast("int")
    )
    .dropDuplicates(["aeronave_id"])
)

# Transformaciones Silver aeropuertos

aeropuertos_slv = (
    aeropuertos_bz
    .select(
        "aeropuerto_id",
        "nombre",
        "ciudad",
        "pais",
        "lat",
        "lon"
    )
    .withColumn("lat", F.col("lat").cast("double"))
    .withColumn("lon", F.col("lon").cast("double"))
    .dropDuplicates(["aeropuerto_id"])
)

# Transformaciones Silver mantenimientos

mantenimientos_slv = (
    mantenimientos_bz
    .select(
        "mantenimiento_id",
        "aeronave_id",
        "fecha",
        "tipo",
        "costo_usd",
        "duracion_hr"
    )
    .withColumn("fecha", F.to_date(F.col("fecha")))
    .withColumn("costo_usd", F.col("costo_usd").cast("double"))
    .withColumn("duracion_hr", F.col("duracion_hr").cast("double"))
    .dropDuplicates(["mantenimiento_id"])
)

# =========================
# 3. Validaciones de calidad
# =========================

nulos_vuelos = vuelos_slv.filter(
    F.col("vuelo_id").isNull()
).count()

duplicados_vuelos = (
    vuelos_slv
    .groupBy("vuelo_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("Nulos vuelo_id:", nulos_vuelos)
print("Duplicados vuelos:", duplicados_vuelos)

if nulos_vuelos > 0:
    raise ValueError(
        "Se encontraron vuelos con identificadores nulos"
    )

if duplicados_vuelos > 0:
    raise ValueError(
        "Se encontraron identificadores de vuelos duplicados"
    )

print("Validaciones de calidad completadas")

# =========================
# 4. Guardar tablas Silver
# =========================

vuelos_slv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.vuelos")

aeronaves_slv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.aeronaves")

aeropuertos_slv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos")

mantenimientos_slv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos")

print("Silver Delta completada correctamente")

print(
    f"Tablas creadas/actualizadas: "
    f"{CATALOG}.{SILVER_SCHEMA}.vuelos, "
    f"{CATALOG}.{SILVER_SCHEMA}.aeronaves, "
    f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos, "
    f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos"
)

# =========================
# 5. Revisión de vuelos
# =========================

vuelos_silver = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.vuelos"
)

print("Vuelos con origen y destino iguales:")

vuelos_silver.filter(
    F.col("origen_id") == F.col("destino_id")
).select(
    "vuelo_id",
    "origen_id",
    "destino_id",
    "estado"
).orderBy("vuelo_id").show(100, truncate=False)

print("Proceso Silver finalizado correctamente")
