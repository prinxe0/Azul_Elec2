from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col

spark = SparkSession.builder.appName("SpotifyPartitionStrategies").getOrCreate()

df = spark.read.csv("spotify_tracks.csv", header=True, inferSchema=True)

print("=== Dataset Preview ===")
df.show(5)

print("=== Dataset Schema ===")
df.printSchema()

print("\n=== Partition Strategy 1: Repartition by Genre ===")

genre_partitioned = df.repartition("genre")

print("Number of partitions:", genre_partitioned.rdd.getNumPartitions())

genre_summary = (
    genre_partitioned
    .filter(col("popularity") > 50)
    .groupBy("genre")
    .agg(
        count("*").alias("total_tracks"),
        avg("popularity").alias("avg_popularity")
    )
    .sort(col("avg_popularity").desc())
)

print("\n=== Genre Popularity Summary ===")
genre_summary.show()

print("\n=== Partition Strategy 2: Repartition by Release Year ===")

year_partitioned = df.repartition("release_year")

print("Number of partitions:", year_partitioned.rdd.getNumPartitions())

year_summary = (
    year_partitioned
    .filter(col("release_year") >= 2015)
    .groupBy("release_year")
    .agg(
        count("*").alias("total_tracks"),
        avg("tempo").alias("avg_tempo")
    )
    .sort("release_year")
)

print("\n=== Release Year Tempo Summary ===")
year_summary.show()


print("\n=== Top Rock Songs by Popularity ===")

rock_songs = (
    df.filter(col("genre") == "rock")
      .select("track_name", "artist_name", "genre", "popularity")
      .sort(col("popularity").desc())
)

rock_songs.show(10)

spark.stop()