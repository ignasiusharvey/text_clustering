# Text Clustering Implementation
Implementation of text clustering using fastText word embedding and K-means algorithm. The dataset can be accessed via **[Kaggle](https://www.kaggle.com/dodyagung/accident).**

You can read detailed explanation of the process, **[here](https://pages.github.com/).**

## Steps to implement text clustering are

**1. Load dependencies**
```
import pandas as pd
import numpy as np

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import fasttext

from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
```

**2. Load and preprocess dataset**
```
def text_preprocess(series, stemmer, stopwords):
    df = series.str.replace("\n\t",  " ")
    df = df.str.replace(r"[^a-zA-Z ]+", "")
    df = df.str.lower()
    df = df.apply(lambda x: ' '.join([stemmer.stem(item) for item in x.split() if item not in stopwords])) 
    return df
    
# Download first from Kaggle    
data = pd.read_csv('twitter_label_manual.csv')

# Get stopwords and create stemmer using Sastrawi
stopwords = StopWordRemoverFactory().get_stop_words()
stemmer = StemmerFactory().create_stemmer()

# Preprocess the sentences
data['processed_text'] = text_preprocess(data['full_text'], stemmer, stopwords)
```

**3. Train word embedding model with fastText**
```
# Build word embedding model and create one more with dim=3 for experimentation
model = fasttext.train_unsupervised('twitter.txt', model='skipgram', dim=100)

# Apply the word embedding model to the sentences
data['vec'] = data['processed_text'].apply(lambda x: model.get_sentence_vector(x))
```

**4. Apply KMeans clustering algorithm**
```
# Elbow Method to define number of k for the clustering
sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(pd.DataFrame(data['vec'].values.tolist()))
    sum_of_squared_distances.append(km.inertia_)

# Plot it
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# K Means clustering
kmeans = KMeans(n_clusters=3)  
kmeans.fit(data['vec'].values.tolist())

# Fit and predict the cluster
data['cluster'] = kmeans.fit_predict(data['vec'].values.tolist())
```

**5. Plot and see results**
```
# Use PCA to reduce the dimensions
pca = PCA(n_components=3)
data['x'] = pca.fit_transform(data['vec'].values.tolist())[:,0]
data['y'] = pca.fit_transform(data['vec'].values.tolist())[:,1]
data['z'] = pca.fit_transform(data['vec'].values.tolist())[:,2]

# Plot in 2D
plt.scatter(data['y'],data['x'], c=data['cluster'], cmap='rainbow')

# Plot in 3D
fig = plt.figure(1, figsize=(10,10))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(data['x'],data['y'],data['z'], c=data['cluster'], cmap='rainbow')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_facecolor('white')
plt.title("Tweet Clustering using K Means", fontsize=14)

# Count flag on each cluster
data.groupby(['cluster'])['is_accident'].value_counts()
```

Also, you can check the output of each step in the Jupyter Notebook file.
