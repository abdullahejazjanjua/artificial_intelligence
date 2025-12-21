import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data.csv")

if data.isnull().sum().sum() > 0:
    cols = data.select_dtypes(include='number').columns.tolist()
    data[cols] = data[cols].fillna(data[cols].mean())
    data.dropna()
    print("Found NULL values...fixing...")

    
if data.duplicated().sum() > 0:
    data.drop_duplicates()
    print("Found duplicate rows...droping...")
    
    
X = data.select_dtypes(include='number')

scaler = StandardScaler()
X = scaler.fit_transform(X)


wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.show()

model = KMeans(n_clusters=5, init='k-means++', random_state=42)
model.fit(X)
clusters = model.fit_predict(X)

data['Cluster'] = clusters

score = silhouette_score(X, clusters)
print(f"Silhouette Score: {score:.4f}")