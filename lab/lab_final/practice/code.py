import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.cluster import KMeans

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier


# 1. Load and Clean
data = pd.read_csv('../train.csv')
data = data.drop("Id", axis=1)
data = data.drop_duplicates()

# Handle missing values
threshold = len(data) * 0.5
data = data.dropna(thresh=threshold, axis=1)
num_cols = data.select_dtypes(include=['number']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
data = data.dropna()

# 2. Split Data
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Separate Preprocessor Logic
categorical_cols = X.select_dtypes(exclude=['number']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train_transformed, y_train)

y_pred = regressor.predict(X_test_transformed)
rmse = root_mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"RMSE: {rmse:.2f}")
print(f"R2-Score: {r2score:.2f}")

y_train_pred = regressor.predict(X_train_transformed)
abs_residuals = np.abs(y_train - y_train_pred)
threshold_90 = np.percentile(abs_residuals, 90)

X_train_processed = X_train.copy()
X_train_processed['Absolute_Error'] = abs_residuals
X_train_processed['Market_Anomaly'] = (abs_residuals >= threshold_90).astype(int)

neighborhood_anomaly_rates = X_train_processed.groupby('Neighborhood')['Market_Anomaly'].mean()
neighborhood_values = data.groupby('Neighborhood')['SalePrice'].median()

struggle_neighborhoods = neighborhood_anomaly_rates.sort_values(ascending=False).head(5).index

avg_market_value = data['SalePrice'].median()
avg_struggle_value = neighborhood_values[struggle_neighborhoods].mean()

print(f"Overall Market Median Price: ${avg_market_value:,.2f}")
print(f"Average Price of 'Struggle' Neighborhoods: ${avg_struggle_value:,.2f}")

if avg_struggle_value > avg_market_value * 1.2:
    print("\nCONCLUSION: The LR model struggles primarily with HIGH-VALUE neighborhoods.")
elif avg_struggle_value < avg_market_value * 0.8:
    print("\nCONCLUSION: The LR model struggles primarily with LOW-VALUE neighborhoods.")
else:
    print("\nCONCLUSION: The model's struggle is spread across mid-range value neighborhoods.")
    
# If the "Struggle" price is much HIGHER than the Market Median: It means your model is failing on the 
# "NBA players" (Luxury Mansions). Linear Regression usually treats every square foot the same, but a 
# square foot in a mansion is often worth way more than a square foot in a standard home.

# If the "Struggle" price is much LOWER than the Market Median: It means your model is failing on the 
# "Jockeys" (Distressed/Cheap homes). The model might think a house is worth $100k based on its size, 
# but because it's in a neighborhood with high crime or bad schools, it's actually only worth $40k.


cluster_features = pd.DataFrame({
    'GrLivArea': X_train['GrLivArea'],
    'SalePrice': y_train
})

scaler_kmeans = StandardScaler()
cluster_scaled = scaler_kmeans.fit_transform(cluster_features)
# X_train_processed = X_train_processed.reset_index(drop=True)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
X_train_processed['Cluster'] = kmeans.fit_predict(cluster_scaled)


sale_price_median = y_train.median()

X_train_processed['is_premium'] = (y_train > sale_price_median).astype(int)
# 1. Prepare Features (X) and Target (y)
# We use the processed features but must drop 'is_premium' from X 
# so the model doesn't "cheat" by looking at the answer.
X_bagging = X_train_processed.drop(columns=['is_premium'])
y_bagging = X_train_processed['is_premium']

# 2. Define the Base Estimator
# We use Entropy as the split criterion and limit depth to prevent overfitting
base_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)

# 3. Initialize and Fit the Bagging Classifier
# Bagging (Bootstrap Aggregating) reduces variance by training multiple trees on random subsets
bagging_model = BaggingClassifier(
    estimator=base_tree, 
    n_estimators=50, 
    random_state=42
)

X_bagging_transformed = preprocessor.fit_transform(X_bagging)

bagging_model.fit(X_bagging_transformed, y_bagging)

# 4. Evaluate on Training Data
y_bagging_pred = bagging_model.predict(X_bagging_transformed)

print("--- Bagging Classifier Results (Predicting 'is_premium') ---")
print(classification_report(y_bagging, y_bagging_pred))



y_rf = X_train_processed['is_premium']

# 2. Initialize and Fit the Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Uses all available CPU cores for speed
)

rf_model.fit(X_bagging_transformed, y_rf)

# 3. Extract Feature Importance
# Get feature names from the preprocessor to match the importance scores
feature_names = preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

# 4. Create a DataFrame for visualization
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("--- Top 10 Most Important Features ---")
print(fi_df.head(10))

print(X_bagging)