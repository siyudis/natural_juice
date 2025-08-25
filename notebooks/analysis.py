# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from textblob import TextBlob  # for sentiment analysis

# Load data
sales = pd.read_csv("data/weekly_sales.csv")
reviews = pd.read_csv("data/reviews.csv")

# Checking datasets
print(sales.head())
print(reviews.head())

# Clean column names for consistency
sales.columns = sales.columns.str.strip().str.lower().str.replace(" ", "_")
reviews.columns = reviews.columns.str.strip().str.lower().str.replace(" ", "_")

# Check for missing values in sales
print(sales.isnull().sum())

# Convert date column
sales['date'] = pd.to_datetime(sales['date'])

# Preview sales info
sales.info()

# Sales Data Analysis

## Total sales
total_sales = sales['price'].sum()
print(f"Total Sales: ₦{total_sales:.2f}")

## Sales by item
sales_by_item = sales.groupby('item')['price'].sum().sort_values(ascending=False)
print(sales_by_item)

# Plot top items
sales_by_item.plot(kind="bar", title="Sales by Item")
plt.ylabel("Total Sales (₦)")
plt.show()

# Aggregate total sales by item
sales_by_item = sales.groupby('item')['price'].sum()

# Plot pie chart
plt.figure(figsize=(8, 8))
sales_by_item.plot.pie(autopct='%1.1f%%', startangle=90, legend=False)
plt.title('Sales Distribution by Item')
plt.ylabel('')  # Hide y-label for pie chart
plt.show()

# Sales trend over time
daily_sales = sales.groupby('date')['price'].sum()
daily_sales.plot(marker='o', title="Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales (₦)")
plt.show()

# Analysis on Customer Reviews

# Customer review function
def get_review(text):
    polarity = TextBlob(str(text)).sentiment.polarity  # Convert to string to avoid errors on NaN
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply to reviews
reviews['sentiment'] = reviews['review_text'].apply(get_review)
print(reviews.head())

# Count reviews
review_counts = reviews['sentiment'].value_counts()
review_counts.plot(kind='bar', title="Customer Reviews")
plt.show()

# Geo-spatial Analysis

## Count unique customers by state/location
customers_by_state = sales.groupby('state')['customer_id'].nunique()
customers_by_state.plot(kind="bar", title="Customers by Location")
plt.show()

# Load Nigeria states GeoJSON
gdf = gpd.read_file("data/geo_ng.json")

# Inspect dataset
print(gdf.info())

# Aggregate sales by state for plotting
sales_by_state = sales.groupby('state')['price'].sum().reset_index()

# Merge sales with GeoJSON on state name
merged = gdf.merge(sales_by_state, left_on="name", right_on="state", how="left")

# Plot map with sales by state
fig, ax = plt.subplots(figsize=(10, 8))

# Base map of Nigeria states
gdf.plot(ax=ax, color="lightgrey", edgecolor="black")

# Plot sales data with color map
merged.dropna(subset=['price']).plot(column="price", ax=ax, cmap="OrRd", legend=True)

# Annotate state names on the map
for idx, row in merged.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        centroid = row['geometry'].centroid
        x, y = centroid.x, centroid.y
        plt.text(x, y, row['name'], fontsize=8, ha="center")
    elif row['geometry'].geom_type == 'MultiPolygon':
        centroid = row['geometry'].centroid
        x, y = centroid.x, centroid.y
        plt.text(x, y, row['name'], fontsize=8, ha="center")

plt.title("Juice Sales by State (Nigeria)", fontsize=14)
plt.axis("off")
plt.show()
