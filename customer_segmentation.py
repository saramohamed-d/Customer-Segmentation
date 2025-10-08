import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("Mall_Customers.csv")

print("data_shape", data.shape)
print(data.head())

print("States:", data[['Annual Income (k$)', 'Spending Score (1-100)']].values)

# plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
# plt.xlabel("Annual Income (K$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("Annual Income \ Spending Score")
# plt.show()

x = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

inertias = []
sil_scores = []
k_range = range(2,11)

for k in k_range:
    Kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = Kmeans.fit_predict(x_scaled)
    inertias.append(Kmeans.inertia_)
    sil_scores.append(silhouette_score(x_scaled, labels))


plt.plot(list(k_range), inertias, marker='o')
plt.xlabel("K")
plt.ylabel("inertias")
plt.title("Elbow Method")
plt.show()

plt.plot(list(k_range), sil_scores, marker='o')
plt.xlabel("K")
plt.ylabel("sil_scores")
plt.title("silhouette Method")
plt.show()

optimal_k = 5
Kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['cluster'] = Kmeans.fit_predict(x_scaled)

centers = scaler.inverse_transform(Kmeans.cluster_centers_)
print("cluster centers:", centers)

plt.figure(figsize=(6,5))
for cluster_label in sorted(data['cluster'].unique()):
    subset = data[data['cluster'] == cluster_label]
    plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], label = f'cluster {cluster_label}')

plt.scatter(centers[:,0], centers[:,1], c= 'black', marker= 'x', s=200, label='centers')
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer segments (KMeans)")
plt.legend()
plt.show()

avg_spending = data.groupby('cluster')['spending score (1-100)'].agg(['mean', 'count'])
print(" Averge spending per cluster:", avg_spending)