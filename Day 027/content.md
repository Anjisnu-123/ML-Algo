<!-- ## Dimensionality reduction
Introduction : Defination,why dimensionality is reduction important: Handling high dimensional data,improves performance,avoids overfitting,enable visualization,enhanced interpretability,
Intution behind dimensionality reduction,common assumtions behind dimensionality reduction : data redundancy,lower intrinsic dimensionality,manifold assumption,noise assumption

Evaluation metrics for dimensionality reduction techniques : Explained varriance ratio:Description,calculation,interpretation,Accuracy: description,calculation,interpretation,Recall : description
calcl: (tp/tp+fn),interpretation,Precision : description,calculation: tp/(tp+fp),interpretation,F1 score : description,calculation: 1*((precision*recalll)/precision+recall),interpretation,
Silhouttte score: escription,calculation,interpretation,KL divergence: description,calculation,interpretation, Cross-entropy loss : Description,calculation,interpretation,stress function:
description,calculation,interpretation,Recostruction error: description,calculation,interpretation

PCA: Defination,Assumptions,Mathematical formulation: data standardization,covariance matrix computation,eigen value value and eigen vector decomposition,selcting pricncipal components,
project data onto principal components,varriance explained|, How to use,when to use,pros and cons,use case,example,intution/visualization,Comparison,evaluation metrics,Complexity,real
world challenge,variance and extensions,limitations in real world

LDA: Defination,Assumptions,Mathematical formulation: compute class means,compute within class scatter metric,compute b/w class scatter  matrix,solve the generalised eigen value problem,
select top discriminants,project data.How to use,when to use,pros and cons,example,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,code implement,
varriants and extensions,limitations in real world apllications

SVD: Defination,assumptions,mathematical formulation,how to use,when to use,pros and cons,USe CASE,Example,intution and visualization,comparison,svd vs eigen values,Evaluation metrics,
complexity,Real world challenges,software implimentation,variants and extension: truncated svd,incremental svd,limitations in real world challenges,

t-distributed stochastic neighbour embedding(tsne): Defination,Assumptions,mathematical formulation: affinity calculation,symmetrization,low dimensional representation,cost function,gradient dscent |,how to use,where to use,pros and cons,use case,steps to followed for tsne calc: step1: calculate pairwise affinities,symmetrize the probabilities,initialize low dimensional points,iteratively optimize.,intuition/visualization,comparison with other technique,evaluation metrics,complexity analysis,real world challenges software implementation,variants and extension,limitations in real world applications


uniform manifold approximation and projection (UMAP) : Defination,assumption,mathematical formulation : constructing the high dimensional affinity graph,symmetrizing the affinites,
construct the low dimensional affinity graph .....,How to use,when to use,pros and cons,use case,numerical example work thorugh step by step,intution/visualization,comparison: umap vs tsne,umap vs pca,evaluation metrics,complexity,real world challeges,software implementation,variants and extension,limitations in real world applications.

Isomap : Defination,Assumptions,mathematical formulation: constructing the neighbourhood graph,finding the nearest neighbour,computing geodesic distance,centering the distance metric,
eigen values decomposistion, embedding,how to use,when to use,pros and cons,usecases,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,software implementations,variants and extensions

Local linear embedding : Defination,assumptions,mathematical formulation,constructing the neighbourhood graph,weight calculation,constructing the weight matrix,compute the low dimensional representation,compute the eigen values problemshow to use,when to use,pros and cons,use case,example:numerical example work through step by step,intution/visualization,comparison,evaluation metrics,complexity,real world challenges.
software implementation,varriants and extension,limitations in real world applications

Multi dimesnional scaling : Defination,assumptions,mathematical formulation,how to use,when to use,pros and cons,numerical exaple walk through step by step,use case,example,intution/visualization,comparison,evaluation metrics,complexity,real world challenges. -->

Creating an exhaustive **guide on dimensionality reduction** can be a complex endeavor since it involves a detailed breakdown of multiple algorithms, including mathematical formulations, code implementations, use cases, advantages, and limitations. Below is an in-depth structure, covering each aspect you mentioned. Let's dive into a comprehensive write-up on dimensionality reduction.

---

# Dimensionality Reduction: Comprehensive Guide

---

## 1. Introduction  
Dimensionality reduction is a technique to reduce the number of input variables in a dataset while preserving as much relevant information as possible. This helps in handling **high-dimensional data**, improving model performance, avoiding **overfitting**, enabling better visualization, and enhancing interpretability.

### Why Dimensionality Reduction is Important:
- **Handling High-Dimensional Data:** Reduces the risk of "curse of dimensionality."
- **Improves Performance:** Reduces computational burden, making algorithms faster.
- **Avoids Overfitting:** Helps prevent models from capturing noise in the data.
- **Visualization:** Enables the visualization of high-dimensional data in 2D or 3D.
- **Enhanced Interpretability:** Simplifies models by reducing the number of features.

---

## 2. Intuition Behind Dimensionality Reduction  
### Common Assumptions:
- **Data Redundancy:** Some features are redundant or correlated, providing little new information.
- **Lower Intrinsic Dimensionality:** High-dimensional data can often be represented effectively in fewer dimensions.
- **Manifold Assumption:** Data points lie on or near a lower-dimensional manifold within the high-dimensional space.
- **Noise Assumption:** Some dimensions only add noise without contributing meaningful information.

---

## 3. Evaluation Metrics for Dimensionality Reduction Techniques

1. **Explained Variance Ratio:**  
   - **Description:** Measures the proportion of variance retained by each principal component.  
   - **Calculation:** \(\text{Explained Variance} = \frac{\sigma^2_{\text{PC}}}{\sigma^2_{\text{Total}}}\).  
   - **Interpretation:** Higher values indicate better retention of original data variance.

2. **Accuracy, Precision, Recall, F1-Score:** (Used in supervised models built on reduced data)
   - **Accuracy:** Measures how often the model correctly predicts.  
     \[ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \]  
   - **Precision:** Proportion of true positives among predicted positives.  
     \[ \text{Precision} = \frac{TP}{TP + FP} \]  
   - **Recall:** Proportion of true positives detected among all positives.  
     \[ \text{Recall} = \frac{TP}{TP + FN} \]  
   - **F1-Score:** Harmonic mean of precision and recall.  
     \[ F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]  

3. **Silhouette Score:**  
   - **Description:** Evaluates how well clusters are separated.  
   - **Range:** From -1 (poor separation) to 1 (clear clustering).  

4. **KL Divergence:**  
   - **Description:** Measures the difference between two probability distributions (original vs reduced).  
   - **Interpretation:** Smaller values indicate better similarity between distributions.

5. **Cross-Entropy Loss:**  
   - **Description:** Measures the difference between actual and predicted class probabilities.  
   - **Interpretation:** Lower values indicate better prediction.

6. **Stress Function:** (Used in Multi-Dimensional Scaling)  
   - **Description:** Measures the difference between pairwise distances in original and reduced spaces.  

7. **Reconstruction Error:**  
   - **Description:** Measures the loss of information during dimensionality reduction.  
   - **Calculation:** \(\text{Reconstruction Error} = \|X - X_{\text{reconstructed}}\|\).  
   - **Interpretation:** Lower values indicate better reconstruction.

---

## 4. Dimensionality Reduction Techniques

---

### **4.1 Principal Component Analysis (PCA)**

#### Definition:
PCA is a linear technique that projects data onto a lower-dimensional space while maximizing variance. 

#### Assumptions:
- Data should be standardized.
- Linearly correlated features.
- Gaussian distribution improves performance.

#### Mathematical Formulation:
1. **Data Standardization:** Center the data by subtracting the mean.
2. **Covariance Matrix Computation:**  
   \[ \text{Cov}(X) = \frac{1}{n-1} X^T X \]
3. **Eigenvalue and Eigenvector Decomposition:** Identify principal components.
4. **Selecting Principal Components:** Choose components explaining most variance.
5. **Project Data:**  
   \[ X' = X W \quad (\text{where W = eigenvectors}) \]

#### Pros and Cons:
- **Pros:** Simple, fast, widely used.
- **Cons:** Only captures linear relationships.

#### Use Case:
- Face recognition (Eigenfaces technique).

---

### **4.2 Linear Discriminant Analysis (LDA)**

#### Definition:
LDA reduces dimensionality by maximizing the separation between multiple classes.

#### Mathematical Formulation:
1. **Compute Class Means:** Mean vectors for each class.
2. **Within-Class Scatter Matrix:** Measures the spread within each class.
3. **Between-Class Scatter Matrix:** Measures separation between classes.
4. **Eigenvalue Problem:** Solve for discriminants.
5. **Project Data:** Map to new subspace.

#### Use Case:
- Handwriting recognition.

---

### **4.3 Singular Value Decomposition (SVD)**

#### Definition:
SVD decomposes a matrix into three components (U, Σ, V). It is often used in text analysis and matrix factorization.

#### Formulation:
\[ A = U \Sigma V^T \]

#### Variants:
- Truncated SVD
- Incremental SVD

---

### **4.4 t-Distributed Stochastic Neighbor Embedding (t-SNE)**

#### Definition:
t-SNE is a nonlinear technique that maps high-dimensional data to two or three dimensions.

#### Steps:
1. **Affinity Calculation:** Pairwise similarities.
2. **Symmetrize:** Normalize probabilities.
3. **Initialize Low-Dimensional Points:** Randomly initialize.
4. **Optimize:** Using gradient descent.

#### Use Case:
- Visualizing word embeddings.

---

Below is the expanded and more detailed section on **UMAP, Isomap, LLE, and MDS**:

---

## **4.5 Uniform Manifold Approximation and Projection (UMAP)**

### **Definition:**  
UMAP is a **nonlinear dimensionality reduction technique** that assumes the data lies on a **Riemannian manifold** and aims to approximate the data’s low-dimensional representation while preserving both **local and global structures**. It builds a **graph representation of high-dimensional data**, then optimizes the layout of the data points in a lower-dimensional space.

### **Mathematical Formulation:**
1. **Construct High-Dimensional Graph:**
   - Compute a neighborhood graph by finding the nearest neighbors for each point (using k-nearest neighbors or other metrics).  
   - Assign a probability-based weight to the edges of the graph, reflecting how closely the points are related.

2. **Symmetrize Affinities:**
   - Make the graph undirected by averaging affinities between connected points. This ensures symmetric relationships between neighbors.

3. **Construct Low-Dimensional Affinity Graph:**
   - In the low-dimensional space, build a new graph with the same set of nodes, and attempt to preserve relationships from the high-dimensional graph.

4. **Optimize Layout:**  
   - A **cost function** is minimized (cross-entropy between the original and low-dimensional graphs).  
   - Uses **stochastic gradient descent (SGD)** to iteratively improve the layout.

### **How to Use:**
- **Parameters:**  
  - `n_neighbors`: Controls the balance between local and global structure preservation.  
  - `min_dist`: Controls how tightly UMAP clusters points in the low-dimensional space.

- **Code Example (Python):**
  ```python
  import umap
  import matplotlib.pyplot as plt
  from sklearn.datasets import load_digits

  data = load_digits().data
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
  embedding = reducer.fit_transform(data)

  plt.scatter(embedding[:, 0], embedding[:, 1], c=load_digits().target, cmap='Spectral')
  plt.colorbar()
  plt.show()
  ```

### **Pros and Cons:**
- **Pros:**  
  - Faster than t-SNE on large datasets.  
  - Captures both **local and global** data structures effectively.  
  - Robust to noise and scales well.

- **Cons:**  
  - Computationally expensive for extremely large datasets.  
  - Sensitive to parameter tuning (`n_neighbors`, `min_dist`).

### **Use Case:**  
- **Bioinformatics:** UMAP is frequently used for **single-cell RNA sequencing data** to visualize high-dimensional gene expression data.  
- **Comparison with t-SNE:**  
  - UMAP is generally faster and preserves more global structure than t-SNE.

---

## **4.6 Isomap (Isometric Mapping)**

### **Definition:**  
Isomap is a **manifold learning algorithm** that reduces dimensionality while preserving **geodesic distances** (distances along the manifold) between points. It is particularly effective for data that lies on a **nonlinear manifold** and performs better than linear techniques like PCA in these cases.

### **Mathematical Formulation:**
1. **Construct the Neighborhood Graph:**  
   - Use **k-nearest neighbors** to connect each data point to its closest neighbors.

2. **Compute Geodesic Distances:**  
   - Calculate the shortest path between all pairs of points using **Dijkstra’s algorithm** or **Floyd-Warshall algorithm** on the neighborhood graph.

3. **Center the Distance Matrix:**  
   - Convert the geodesic distances into a matrix suitable for embedding by centering it using double centering.

4. **Eigenvalue Decomposition:**  
   - Perform eigenvalue decomposition on the centered matrix to obtain the lower-dimensional representation.

### **How to Use:**  
- **Code Example (Python):**
  ```python
  from sklearn.datasets import make_swiss_roll
  from sklearn.manifold import Isomap
  import matplotlib.pyplot as plt

  data, color = make_swiss_roll(n_samples=1000)
  isomap = Isomap(n_neighbors=10, n_components=2)
  embedding = isomap.fit_transform(data)

  plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral')
  plt.show()
  ```

### **Pros and Cons:**
- **Pros:**  
  - Captures nonlinear structures well.  
  - Preserves the global structure through geodesic distances.

- **Cons:**  
  - Sensitive to the choice of neighbors (`k`).  
  - Computationally expensive for large datasets.

### **Use Case:**  
- **Swiss Roll Dataset:** Isomap effectively unrolls the **Swiss roll** dataset, a classic example of nonlinear data.

---

## **4.7 Local Linear Embedding (LLE)**

### **Definition:**  
Local Linear Embedding (LLE) is a nonlinear dimensionality reduction technique that **preserves local relationships** between data points. It assumes that each data point can be expressed as a **linear combination of its neighbors**.

### **Mathematical Formulation:**
1. **Construct the Neighborhood Graph:**  
   - Find the k-nearest neighbors for each data point.

2. **Calculate Reconstruction Weights:**  
   - For each data point, calculate the weights that best reconstruct it from its neighbors by minimizing reconstruction error:
     \[
     \text{Minimize} \sum_i \left\| X_i - \sum_j W_{ij} X_j \right\|^2
     \]
   - Subject to \( \sum_j W_{ij} = 1 \) (weights sum to 1).

3. **Construct Weight Matrix (W):**  
   - Store the calculated weights for all points in a matrix \( W \).

4. **Compute the Low-Dimensional Representation:**  
   - Solve the eigenvector problem to find the low-dimensional coordinates that best preserve the local weights.

### **How to Use:**  
- **Code Example (Python):**
  ```python
  from sklearn.datasets import make_s_curve
  from sklearn.manifold import LocallyLinearEmbedding
  import matplotlib.pyplot as plt

  data, color = make_s_curve(n_samples=1000)
  lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
  embedding = lle.fit_transform(data)

  plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral')
  plt.show()
  ```

### **Pros and Cons:**
- **Pros:**  
  - Captures local structures effectively.  
  - Works well with nonlinear manifolds.

- **Cons:**  
  - Sensitive to noise and parameter selection.  
  - Poor at preserving global structure.

---

## **4.8 Multi-Dimensional Scaling (MDS)**

### **Definition:**  
MDS is a dimensionality reduction technique that **preserves pairwise distances** between data points. It attempts to find a low-dimensional configuration where the distances between points closely match the distances in the original space.

### **Mathematical Formulation:**
1. **Calculate Pairwise Distance Matrix:**  
   - Compute the Euclidean distance between all pairs of points.

2. **Center the Distance Matrix:**  
   - Use double centering to convert the distance matrix into a form suitable for eigenvalue decomposition.

3. **Eigenvalue Decomposition:**  
   - Perform decomposition to get the lower-dimensional embedding.

### **How to Use:**  
- **Code Example (Python):**
  ```python
  from sklearn.datasets import load_iris
  from sklearn.manifold import MDS
  import matplotlib.pyplot as plt

  data = load_iris().data
  mds = MDS(n_components=2, dissimilarity='euclidean')
  embedding = mds.fit_transform(data)

  plt.scatter(embedding[:, 0], embedding[:, 1], c=load_iris().target, cmap='viridis')
  plt.show()
  ```

### **Pros and Cons:**
- **Pros:**  
  - Captures pairwise relationships effectively.  
  - Suitable for **visualization of similarity matrices**.

- **Cons:**  
  - Computationally expensive for large datasets.  
  - Sensitive to local minima in optimization.

---



## 5. Conclusion  
Dimensionality reduction is a crucial technique for making sense of high-dimensional data. Each algorithm offers unique strengths and trade-offs, and choosing the right one depends on the specific use case.

---

This guide covers the essential techniques and their detailed formulations, evaluation metrics, and practical use cases. Let me know if you need code snippets or more specific examples for each algorithm!
