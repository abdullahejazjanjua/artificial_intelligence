import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("heart.csv")

if data.isnull().sum().sum() > 0:
    cols = data.select_dtypes(include="number").columns
    data[cols] = data[cols].fillna(data[cols].mean())
    data = data.dropna()
    print("Found NULL values...fixing...")

if data.duplicated().sum() > 0:
    data = data.drop_duplicates()
    print("Found duplicate rows...droping...")
    
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

columns = X.columns.tolist()

X_temp = X.copy()
for col in X.select_dtypes(exclude="number"):
    X_temp[col], _ = X_temp[col].factorize()
mi_scores = mutual_info_classif(X_temp, Y)

columns_to_drop = []

for column, mi_score in zip(columns, mi_scores):
    if mi_score == 0.0:
        columns_to_drop.append(column)

X = X.drop(columns_to_drop, axis=1)
columns = [c for c in columns if c not in columns_to_drop]

print(f"Keeping {columns}")
num_cols = 2
num_rows = (len(columns) + 1) // num_cols

fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 15))
ax = ax.flatten()

for idx, column in enumerate(columns):
    ax[idx].hist(X[column], bins=10, edgecolor='black')
    ax[idx].set_title(f"Histogram of {column}")
    ax[idx].grid(False)
    
for idx in range(len(columns), len(ax)):
    ax[idx].axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 15))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f")
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


model = Pipeline([
    ('pre', preprocessor),
    ('poly', PolynomialFeatures(degree=1)),
    # ('classify', LogisticRegression())
    
    # ('classify', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    
    # ('classify', DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42))
    
    ('classify',  KNeighborsClassifier(n_neighbors=5)) 
    
    # ('classify', BaggingClassifier(
    #         estimator=DecisionTreeClassifier(max_depth=5), 
    #         n_estimators=50,                             
    #         max_samples=0.8,                             
    #         max_features=1.0,                            
    #         random_state=42
    #     ))
    
    # ('classfiy', VotingClassifier(
    #     estimators=[
    #         ('m1', LogisticRegression()),
    #         ('m2', DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)),
    #         ('m3', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    #         ('m4', KNeighborsClassifier(n_neighbors=5))
    #     ]
    # ))
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(classification_report(Y_test, y_pred))


cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()

# param_grid = {
#     'classfiy__m2__max_depth': [3, 5, 10],
#     'classfiy__m2__min_samples_split': [2, 5, 10],
#     'classfiy__m3__n_estimators': [100, 50, 10],
#     'classfiy__m3__max_depth': [3, 5, 10],

# }

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train, Y_train)

# print(f"Best Parameters: {grid_search.best_params_}")


# Recall: Out of all the people who actually had heart disease, how many did we catch?
# Precision: Out of all the people the model predicted as having heart disease, how many actually had it?
# Confusion Matrix:
    # [ TN , FP ] 
    # [ FN , TP ]