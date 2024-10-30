import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reads excel file 
df = pd.read_excel('Test_data.xlsx')
x = df.iloc[:, 0] # Adjust to the desired X variable as needed
y = df.iloc[:, 1] # Adjust to the desired Y variable as needed

# Creates 'Inertia' for elbow method 
data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Adds extra columns to the 'Inertia' dataframe to facilitate estimation computations
df2 = pd.DataFrame(inertias, columns=['Inertias'])
df2.insert(1, 'nCluster_Elbow', np.nan)
df2['Inertias+1'] = [0] + df2['Inertias'].tolist()[:-1]
df2.insert(1, 'Inertias+1', df2.pop('Inertias+1'))

# Computation for K-value estimation
df2['nCluster_Elbow'] = (df2['Inertias+1'] / df2['Inertias'])
df2['nCluster_Elbow_S'] = [0] + df2['nCluster_Elbow'].tolist()[:-1]
df2['nCluster_Elbow_D'] = df2['nCluster_Elbow_S'] - df2['nCluster_Elbow']
df2['n_cluster'] = range(1, len(df2) + 1)

# Locates estimation computation results in the pandas dataframe and prints in console
Max_nCluster_Elbow_D = df2['nCluster_Elbow_D'].idxmax()
calculated_k = df2.iloc[Max_nCluster_Elbow_D]['n_cluster']
print("The estimated number of clusters for this project:", calculated_k)


# Scikit-learn K-Means clustering model
calculated_k = int(calculated_k)
kmeans = KMeans(n_clusters=calculated_k)
kmeans.fit(data)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-Means Clustering Plot')
plt.rcParams["figure.figsize"] = (10,5)

plt.scatter(x, y, c=kmeans.labels_)
    
plt.savefig('output.png')
plt.show()


