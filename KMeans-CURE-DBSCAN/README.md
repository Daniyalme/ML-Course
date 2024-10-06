# Clustering Algorithms: CURE, K-Means, DBSCAN, and Image Compression

This folder contains four machine learning projects focused on different clustering algorithms, including CURE, K-Means, and DBSCAN. These projects cover a wide range of applications such as breast cancer diagnosis, image compression, and clustering visualizations on different datasets. The projects aim to explore the effectiveness of these clustering techniques in various tasks.

---

## 1. CURE Algorithm for Clustering and Comparison with Hierarchical Clustering

### Overview
This project implements the CURE (Clustering Using Representatives) algorithm to cluster data and compare its performance with hierarchical clustering. The CURE algorithm is effective for handling large datasets and noise, making it suitable for various real-world applications.

### Key Implementation Steps
- Implement the CURE algorithm to cluster data points.
- Compare the clustering results with hierarchical clustering to analyze differences in performance.
- Visualize the clustering outcomes.

---

## 2. Breast Cancer Prevention Using K-Means Algorithm

### Overview
This project applies the K-Means clustering algorithm to classify breast cancer cells based on the Wisconsin Diagnostic Breast Cancer dataset. The dataset contains 30 features extracted from cell images, and the goal is to cluster the data for cancer detection.


### Key Implementation Steps
- Implement a Python class for K-Means clustering.
- Define functions for fitting the model, calculating accuracy, predicting cluster labels, and computing the sum of squared errors (SSE).
- Run the K-Means algorithm with different initializations and observe the results.

---

## 3. Image Compression using K-Means

### Overview
This project uses the K-Means algorithm for image compression. The algorithm computes centroids for pixel clusters, and each pixel is represented by the nearest centroid, reducing the image size without losing much detail.

### Key Implementation Steps
- Apply the K-Means algorithm to compress images by calculating centroids for pixel clusters.
- Represent each pixel by its nearest centroid to compress the image.
- Compare the original and compressed images to evaluate the compression quality.

---

## 4. DBSCAN Clustering and Visualization on Different Data Shapes

### Overview
This project implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm from scratch and visualizes its performance on datasets with different shapes. DBSCAN is particularly effective for identifying clusters of arbitrary shapes and handling noise.

### Key Implementation Steps
- Implement the DBSCAN algorithm on datasets with different geometric shapes.
- Visualize the clustering results to observe how DBSCAN identifies clusters of arbitrary shapes.
- Adjust DBSCAN parameters such as `eps` and `min_samples` to improve clustering performance.

---

## General Guide to Setting Up Datasets

1. **Extract the downloaded datasets**: Extract the downloaded dataset to the specified folder inside the project directory if needed.
2. **Check file paths**: Verify that the file paths in the code match the location of the datasets or images.
3. **Install necessary libraries**: Ensure you have installed all the required Python libraries, such as `scikit-learn`, `numpy`, `matplotlib`, and `pandas`.
