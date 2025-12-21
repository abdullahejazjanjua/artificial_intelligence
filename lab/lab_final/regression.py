import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

data = pd.read_csv("data.csv")

if data.isnull().sum().sum() > 0:
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    data = data.dropna()
    print("NULL values found...fixing...")
   
if data.duplicated().sum() > 0:
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    print("Found duplicate rows...dropping...")


X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
columns = X.columns.tolist()


temp_data = X.copy()
for col in temp_data.select_dtypes(exclude="number"):
    temp_data[col], _ = temp_data[col].factorize()
mi_scores = mutual_info_regression(temp_data, Y)

columns_to_drop = []
for column, mi_score in zip(columns, mi_scores):
    if mi_score < 0.2:
        columns_to_drop.append(column)

X = X.drop(columns_to_drop, axis=1)
columns = [c for c in columns if c not in columns_to_drop]

num_cols = 2
num_rows = (len(X.columns) + 1) // num_cols

fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5))
ax = ax.flatten()

for idx, column in enumerate(columns):
    ax[idx].hist(data[column], bins=10, edgecolor="black")
    ax[idx].set_title(f"Histogram of {column}")
    ax[idx].grid(False)
plt.tight_layout()
plt.show()

corr  = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5))
ax = ax.flatten()

for idx, column in enumerate(columns):
    ax[idx].scatter(data[column], Y)
    ax[idx].set_title(f"Scatter of {column} VS {Y.name}")
    ax[idx].grid(False)
    
plt.tight_layout()
plt.show()

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ]
)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = Pipeline([
    ('pre', preprocessor),
    ('poly', PolynomialFeatures(degree=1)),
    # ('regression', LinearRegression())
    
    # ('regression', DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)),
    
    # ('regression', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    
    # ('regression', BaggingRegressor(
    #         estimator=DecisionTreeRegressor(max_depth=5), # The base model to "bag"
    #         n_estimators=50,                             # Number of models to train
    #         max_samples=0.8,                             # Each model sees 80% of data
    #         max_features=1.0,                            # Each model sees all columns
    #         random_state=42
    #     ))
    
    ('regression', VotingRegressor(
        estimators=[
            ('m1', LinearRegression()),
            ('m2', DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)),
            ('m3', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ]
    ))
])

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(f"RMSE: {root_mean_squared_error(Y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(Y_test, y_pred):.2f}")

# param_grid = {
#     'regression__m2__max_depth': [3, 5, 10],
#     'regression__m2__min_samples_split': [2, 5, 10],
#     'regression__m3__n_estimators': [100, 50, 10],
#     'regression__m3__max_depth': [3, 5, 10],

# }

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train, Y_train)

# print(f"Best Parameters: {grid_search.best_params_}")