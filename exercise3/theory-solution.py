#!/usr/bin/env python
# coding: utf-8

# 1) What are the advantages of t-SNE over PCA?
# 
# - t-SNE is better at capturing non-linear relationships between data points, while PCA assumes that the relationships are linear.
# 
# - t-SNE can preserve the local structure of the data, which is useful when analyzing complex datasets with many clusters or subgroups.
# 
# - t-SNE produces a probability distribution that can be used to measure the similarity between data points, while PCA only produces a linear transformation of the data.

# In[29]:


import numpy as np
from scipy.spatial.distance import pdist, squareform


# In[39]:


def calculate_pairwise_similarities(x, y):
    """
    Calculates the pairwise similarities for P and Q given the high-dimensional
    and low-dimensional embeddings, respectively.
    """
    # Compute pairwise distances
    distances_x = squareform(pdist(x, 'euclidean'))
    distances_y = squareform(pdist(y, 'euclidean'))
    
    # Compute Gaussian similarities for P
    variance_x = np.var(distances_x)
    P = np.exp(-distances_x ** 2 / (2 * variance_x))
    np.fill_diagonal(P, 0)
    P /= np.sum(P)
    
    # Compute Student t-distribution similarities for Q
    # Picked this due to crowding problem for Gaussian
     
    dof = 1  # Degrees of freedom for Student t-distribution
    t = 1 / (1 + (distances_y ** 2 / dof))
    np.fill_diagonal(t, 0)
    Q = t / np.sum(t)
    return P, Q


def kl_divergence(prob_dist_1, prob_dist_2, eps=1e-10):
    prob_dist_2 = np.clip(prob_dist_2, eps, None)  # Clip Q to avoid log(0)
    prob_dist_1 = np.clip(prob_dist_1, eps, None)  # Clip P to avoid log(0)
    return np.sum(prob_dist_1 * np.log(prob_dist_1 / prob_dist_2))


# In[63]:


# Scenario 1: a, b, and c are all close to each other in the low-dimensional space, 
# but a and b are very close to each other in the high-dimensional space while c is far away

x1 = np.array([[1, 2, 3], [2, 3, 4], [100, 120, 130]])
y1 = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])

p1, q1 = calculate_pairwise_similarities(x1, y1)
loss1 = kl_divergence(p1, q1)
print("Scenario 1 loss:", loss1)


# In the first scenario where a, b, and c are all close to each other in the low-dimensional space, the pairwise similarities between them will be high, since they are close to each other. However, the pairwise similarities between a and b in the high-dimensional space were very high, while the similarity between c and either a or b was very low. As a result, the loss function will be high because there is a mismatch between the pairwise similarities in the high-dimensional space and the low-dimensional space.

# In[58]:


# Scenario 2: a and b are close to each other in the low-dimensional space, but c is far away from them
x2 = np.array([[1, 2, 3], [2, 3, 4], [100, 120, 130]])
y2 = np.array([[0.1, 0.2], [0.2, 0.3], [55, 45]])

p2, q2 = calculate_pairwise_similarities(x2, y2)
loss2 = kl_divergence(p2, q2)
print("Scenario 2 loss:", loss2)


# The KL divergence is likely to be relatively low, indicating that the t-SNE algorithm is able to effectively represent the data in the low-dimensional space,since the pairwise similarities between a and b are high in both the high-dimensional and low-dimensional spaces, and the pairwise similarities between c and a (or c and b) are low.

# In[60]:


# Scenario 3: a, b, and c are all far away from each other in the low-dimensional space
x3 = np.array([[1, 2, 3], [2, 3, 4], [100, 120, 130]])
y3 = np.array([[0.1, 0.2], [500.9, 500.8], [1000.5, 1000.6]])

p3, q3 = calculate_pairwise_similarities (x3, y3)
loss3 = kl_divergence(p3, q3)
print("Scenario 3 loss:", loss3)


# Since the pairwise similarities between a, b, and c are low in the low-dimensional space and the pairwise similarities between a and b are high in the high-dimensional space, the KL divergence is likely to be relatively high, indicating that the t-SNE algorithm may have difficulty representing the data in the low-dimensional space.

# In[61]:


# Scenario 4: a is far away from both b and c, which are close to each other in the low-dimensional space
x4 = np.array([[1, 2, 3], [2, 3, 4], [100, 120, 130]])
y4 = np.array([[0.1, 0.2], [58.8, 58.9], [58.9, 60.0]])

p4, q4 = calculate_pairwise_similarities(x4, y4)
loss4 = kl_divergence(p4, q4)
print("Scenario 4 loss:", loss4)


# Since the pairwise similarities between a, b, and c are low in the low-dimensional space and the pairwise similarities between b and c are high in the high-dimensional space, the KL divergence is likely to be relatively high, indicating that the t-SNE algorithm may have difficulty representing the data in the low-dimensional space.
