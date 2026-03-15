from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Flight Delay Analysis") \
    .config("spark.driver.extraJavaOptions","--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
    .getOrCreate()

print("Spark Session Started")

df = spark.read.csv(
    "departuredelays.csv",  
    header=True,
    inferSchema=True
)

print("Dataset Loaded")

df.show(5)

print("Dataset Schema:")
df.printSchema()

print("Selected Columns:")
df.select("origin","destination","delay").show()

print("Flights with Delay > 10 minutes:")
df.filter(df.delay > 10).show()

df_clean = df.dropna().dropDuplicates()

print("Cleaned Data:")
df_clean.show(5)

df_clean.createOrReplaceTempView("flights")

print("Top 5 Most Frequent Destinations:")

top_destinations = spark.sql("""
SELECT destination, COUNT(*) AS total_flights
FROM flights
GROUP BY destination
ORDER BY total_flights DESC
LIMIT 5
""")

top_destinations.show()

print("Top 5 Origins with Highest Average Delay:")

spark.sql("""
SELECT origin, AVG(delay) AS avg_delay
FROM flights
GROUP BY origin
ORDER BY avg_delay DESC
LIMIT 5
""").show()

spark.stop()

print("Spark Session Stopped")