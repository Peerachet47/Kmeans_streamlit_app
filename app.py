#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:14:26 2025

@author: peerachet
"""
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#load models
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
#set the title
st.title("K-MEANS Clustering Visualizer by Peerachet Khanitson 6531501092")

#set the page config
st.set_page_config(page_title = "K-means Clustering", layout = "centered")

#load datasets
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predict using the loaded model
y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)