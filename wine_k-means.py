# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set the styles to Seaborn
sns.set()
# Import the KMeans module so we can perform k-means clustering with sklearn
from sklearn.cluster import KMeans
# veriyi al
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', \
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
              'Proline']
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = wine_names) 
wine_data['Class']=wine_data['Class']-1
data = pd.DataFrame(wine_data)
x=data[['Alcohol','OD280/OD315']]
# iki değişkenin saçılım grafiğini oluştur
plt.scatter(x['Alcohol'],x['OD280/OD315'])
# eksenleri isimlendir
plt.xlabel('Alkol')
plt.ylabel('OD280/OD315')
plt.show()
# k sayısını belirleyebilmek içim elbow metodundan faydalan
wcss = []
# 'cl_num', WCSS yöntemini kullanmak istediğimiz en fazla kümeyi izleyen bir dizidir. 10'a ayarladık, ama tamamen keyfi.
cl_num = 10
for i in range (1,cl_num):
    kmeans= KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
number_clusters = range(1,cl_num)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()
# k'yi 3 olarak seç ve devam et
kmeans_3 = KMeans(3,random_state=50)
kmeans_3.fit(x)
# girdi verisinin kopyasını yarat
clusters_3 = x.copy()
# Öngörülen kümeleri not et
clusters_3['cluster_pred']=kmeans_3.fit_predict(x)
# Verileri çiz
# c (renk) bir değişkenle kodlanabilen bir bağımsız değişkendir
# Bizim durumumuzda değişken, plt.scatter'a üç renk olduğunu gösteren 0,1,2 değerlerine sahiptir
# Küme 0, 1 ve 2'deki tüm noktalar aynı renk
# cmap renk haritasıdır. rainbow güzeldir, ama jet'i de dene
plt.scatter(clusters_3['Alcohol'], clusters_3['OD280/OD315'], c= clusters_3 ['cluster_pred'], cmap = 'rainbow')
plt.show()
# doğruluğu hesaplayacağımız veri çerçevesini oluştur 
dogruluk = data[['Class', 'Alcohol', 'OD280/OD315']]
# veri çerçevemize tahinleri yeni bir kolon olarak ekle
dogruluk['tahmin']=clusters_3['cluster_pred']
# tahminle orjinal veri setindeki sınıf aynı olup olmadığını karşılaştıran yeni bir mantıksal kolon ekle
dogruluk['aynimi']=dogruluk['tahmin']==dogruluk['Class']
# başarı sayısını ve oranını yazdır
print(dogruluk['aynimi'].value_counts())
print(dogruluk['aynimi'].value_counts('%'))
print(clusters_3['cluster_pred'].head())
print(data['Class'].head())