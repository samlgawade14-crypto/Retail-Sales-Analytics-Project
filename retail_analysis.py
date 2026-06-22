import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/Superstore.csv", encoding="latin1")

# Convert Date Columns
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Professional Theme
sns.set_theme(style="whitegrid")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==================================
# CHART 1 - SALES BY CATEGORY
# ==================================

sales_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))

bars = plt.bar(
    sales_category.index,
    sales_category.values,
    color=["#4E79A7", "#F28E2B", "#59A14F"]
)

plt.title("Sales Performance by Category", fontsize=18, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Sales ($)")

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"${y:,.0f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("images/sales_by_category.png", dpi=300)
plt.show()

# ==================================
# CHART 2 - PROFIT BY CATEGORY
# ==================================

profit_category = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))

bars = plt.bar(
    profit_category.index,
    profit_category.values,
    color=["#2E8B57", "#FF8C00", "#4682B4"]
)

plt.title("Profit by Category", fontsize=18, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Profit ($)")

plt.tight_layout()
plt.savefig("images/profit_by_category.png", dpi=300)
plt.show()

# ==================================
# CHART 3 - MONTHLY SALES TREND
# ==================================

monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()

monthly_sales.index = monthly_sales.index.astype(str)

plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=3
)

plt.title("Monthly Sales Trend", fontsize=18, fontweight="bold")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig("images/monthly_sales_trend.png", dpi=300)
plt.show()

# ==================================
# CHART 4 - REGION SALES
# ==================================

region_sales = df.groupby("Region")["Sales"].sum().sort_values()

plt.figure(figsize=(10,6))

plt.barh(
    region_sales.index,
    region_sales.values,
    color=["#6A5ACD", "#20B2AA", "#FFA500", "#FF6347"]
)

plt.title("Sales by Region", fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig("images/region_sales.png", dpi=300)
plt.show()

# ==================================
# CHART - TOP 10 STATES BY SALES
# ==================================

top_states = (
    df.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

bars = plt.bar(
    top_states.index,
    top_states.values,
    color=[
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC"
    ]
)

plt.title(
    "Top 10 States by Sales",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("State")
plt.ylabel("Sales ($)")

plt.xticks(rotation=45)

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"${y:,.0f}",
        ha="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "images/top10_states_sales.png",
    dpi=300
)

plt.show()

# ==================================
# CHART 5 - CORRELATION HEATMAP
# ==================================

plt.figure(figsize=(8,6))

corr = df[["Sales","Profit","Discount","Quantity"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=1
)

plt.title("Correlation Heatmap", fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.show()

print("\nAll charts generated successfully!")

# ==================================
# SALES PREDICTION MODEL
# ==================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features
X = df[["Quantity", "Discount", "Profit"]]

# Target
y = df["Sales"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print("R2 Score :", round(r2_score(y_test, y_pred), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE     :", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))