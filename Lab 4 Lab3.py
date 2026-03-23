from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Flight Delay Analysis with Visualization") \
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
df.printSchema()
df_clean = df.dropna().dropDuplicates()
df_clean.createOrReplaceTempView("flights")


top_destinations = spark.sql("""
SELECT destination, COUNT(*) AS total_flights
FROM flights
GROUP BY destination
ORDER BY total_flights DESC
LIMIT 5
""")

avg_delay = spark.sql("""
SELECT origin, AVG(delay) AS avg_delay
FROM flights
GROUP BY origin
ORDER BY avg_delay DESC
LIMIT 10
""")

delay_dist = df_clean.select("delay")

origin_counts = spark.sql("""
SELECT origin, COUNT(*) AS total_flights
FROM flights
GROUP BY origin
ORDER BY total_flights DESC
LIMIT 10
""")

route_data = spark.sql("""
SELECT origin, destination, COUNT(*) AS total
FROM flights
GROUP BY origin, destination
ORDER BY total DESC
LIMIT 10
""")

top_dest_pd = top_destinations.toPandas()
avg_delay_pd = avg_delay.toPandas()
delay_dist_pd = delay_dist.toPandas()
origin_counts_pd = origin_counts.toPandas()
route_pd = route_data.toPandas()

plt.figure()
plt.bar(top_dest_pd['destination'], top_dest_pd['total_flights'])
plt.title("Top 5 Destinations")
plt.xlabel("Destination")
plt.ylabel("Flights")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.bar(avg_delay_pd['origin'], avg_delay_pd['avg_delay'])
plt.title("Average Delay per Origin")
plt.xlabel("Origin")
plt.ylabel("Avg Delay")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.hist(delay_dist_pd['delay'], bins=20)
plt.title("Delay Distribution")
plt.xlabel("Delay")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.plot(origin_counts_pd['origin'], origin_counts_pd['total_flights'], marker='o')
plt.title("Top Origins by Flight Count")
plt.xlabel("Origin")
plt.ylabel("Flights")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.scatter(avg_delay_pd['origin'], avg_delay_pd['avg_delay'])
plt.title("Origin vs Avg Delay")
plt.xlabel("Origin")
plt.ylabel("Avg Delay")
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.barplot(data=top_dest_pd, x='destination', y='total_flights', palette='viridis')
plt.title("Top Destinations")
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.barplot(data=avg_delay_pd, x='origin', y='avg_delay', palette='coolwarm')
plt.title("Average Delay per Origin")
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.histplot(delay_dist_pd['delay'], bins=20, kde=True, color='blue')
plt.title("Delay Distribution with KDE")
plt.show()

plt.figure()
sns.boxplot(y=delay_dist_pd['delay'], color='orange')
plt.title("Boxplot of Delays")
plt.show()

route_pivot = route_pd.pivot(index='origin', columns='destination', values='total')

plt.figure()
sns.heatmap(route_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Route Frequency Heatmap")
plt.show()

spark.stop()
print("Spark Session Stopped")