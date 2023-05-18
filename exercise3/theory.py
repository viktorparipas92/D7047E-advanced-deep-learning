from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE


first_point = [0, 0, 0]
second_point = [5, 5, 5]
third_point = [100, 100, 100]
cluster_centers = [first_point, second_point, third_point]

stdev = 0.1
cluster_size = 10
num_dimensions = len(first_point)


points = np.empty((0, 3))
for cluster_center in cluster_centers:
    new_points = np.random.normal(
        loc=cluster_center, scale=stdev, size=(cluster_size, num_dimensions)
    )
    points = np.concatenate((points, new_points), axis=0)


# perplexities = range(2, 21)
perplexities = [2, 20]

for perplexity in perplexities:
    tsne = TSNE(perplexity=perplexity)
    embedded_features = tsne.fit_transform(points)
    plt.scatter(*embedded_features.T)
    plt.show()
    print(f"Perplexity {perplexity}: {tsne.kl_divergence_}")


# Define the 3 points in 3D space
x1 = np.array([1, 1, 1])
x2 = np.array([2, 2, 2])
x3 = np.array([10, 10, 10])

# Define the 2D mapping of the 3 points
y1 = np.array([1, 1])
y2 = np.array([2, 2])
y3 = np.array([10, 10])

# Combine the points into arrays
x = np.vstack([x1, x2, x3])
y = np.vstack([y1, y2, y3])


def get_gaussian_similarities(distances):
    return np.exp(-distances ** 2 / (2 * np.median(distances) ** 2))


def get_student_t_similarities(distances):
    return 1 / (1 + distances ** 2 / np.median(distances) ** 2)


def calculate_kl_divergence(
    original_points,
    mapped_points,
    mapped_similarity='gaussian',
):
    original_distances = squareform(pdist(original_points))
    mapped_distances = squareform(pdist(mapped_points))

    original_similarities = get_gaussian_similarities(original_distances)
    if mapped_similarity == 'gaussian':
        mapped_similarities = get_gaussian_similarities(mapped_distances)
    elif mapped_similarity == 'student-t':
        mapped_similarities = get_student_t_similarities(mapped_distances)
    else:
        raise Exception

    return np.sum(
        original_similarities
        * np.log(original_similarities / mapped_similarities)
    )


mappings = [
    [[1, 1], [2, 2], [3, 3]],
    [[1, 1], [2, 2], [10, 10]],
    [[1, 1], [5.5, 5.5], [10, 10]],
    [[1, 1], [9, 9], [10, 10]],
]
for i, mapping in enumerate(mappings):
    kl = calculate_kl_divergence(
        original_points=np.vstack([x1, x2, x3]),
        mapped_points=np.array(mapping),
        # mapped_similarity='student-t',
        mapped_similarity='gaussian',
    )
    print(f"Scenario {i}: {kl}")