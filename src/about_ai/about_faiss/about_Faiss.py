from typing import List
import faiss
import numpy as np
from sympy import primerange

INDEXES = {"d1_index": faiss.IndexFlatL2(1), "d2_index": faiss.IndexFlatL2(2)}


def get_accuracy(vectors, index):
    k = 1
    good: int = 0
    bad: int = 0
    bad_children = []
    # _distances, _instances = INDEXES[index].search(vectors, k=k)
    # instances = [x.item() for x in _instances]
    # distances = [x.item() for x in _distances]

    for i, vector in enumerate(vectors):
        query = np.array([vector], dtype=np.float32)
        _distances, _instances = INDEXES[index].search(query, k=k)

        instance = _instances[0][0]
        distance = _distances[0][0]

        if instance == i:
            good += 1
        else:
            bad += 1
            bad_children.append((i, distance, vectors[i], vectors[instance]))

    return (good / (good + bad), good, bad, bad_children)


if __name__ == "__main__":
    prime_numbers = list(primerange(2, 1_000_000))
    print(
        f"prime_numbers: {len(prime_numbers)}, first: {prime_numbers[0]}, last: {prime_numbers[-1]}"
    )

    # single dimension vectors
    d1_vectors = np.array([[p] for p in prime_numbers], dtype=np.float32)
    print(
        f"Shape: {d1_vectors.shape}; dimensions: {d1_vectors.ndim}; Size: {d1_vectors.size}"
    )

    INDEXES["d1_index"].add(d1_vectors)
    if len(prime_numbers) != INDEXES["d1_index"].ntotal:
        raise Exception("[ERROR] Index not complete")
    accuracy, good, bad, bad_children = get_accuracy(d1_vectors, "d1_index")
    if bad_children:
        max_distance = np.max(
            [c[1] for c in bad_children]
        )  # max(c[1] for c in bad_children)
        min_distance = np.min(
            [c[1] for c in bad_children]
        )  # min(c[1] for c in bad_children)
        print(
            f"Unidimensional vectors => Accuracy: {accuracy:6.1%} | good: {good:>6,} / {good+bad:>6,} | bad {bad:>6,} / {good+bad:>6,}"
        )
        for bad_child in bad_children:
            print(f"    {bad_child}")
    else:
        print(
            f"Unidimensional vectors => Accuracy: {accuracy:6.1%} | good: {good:>6,} / {good+bad:>6,} | bad {bad:>6,} / {good+bad:>6,}"
        )

    print()

    # bi-dimensional vectors
    # Let's create a set of vectors that point to points in a circumference of radius 20 := x² + y² = 400
    #   x in [-20, 20]
    #   y in [-20, 20]

    points_2d: List[List[float]] = []

    x: float = 0.0
    delta: float = 0.001
    radius = 20.0
    while (radius - x) >= 0.0:
        y = np.sqrt(400 - x * x)
        points_2d += [[x, y], [-x, y], [x, -y], [-x, -y]]
        x += delta

    bidim = 2
    d2_vectors = np.array(points_2d, dtype=np.float32)
    print(
        f"Shape: {d2_vectors.shape}; dimensions: {d2_vectors.ndim}; Size: {d2_vectors.size}"
    )

    INDEXES["d2_index"].add(d2_vectors)
    accuracy, good, bad, bad_children = get_accuracy(d2_vectors, "d2_index")
    if bad_children:
        max_distance = np.max(
            [c[1] for c in bad_children]
        )  # max(c[1] for c in bad_children)
        min_distance = np.min(
            [c[1] for c in bad_children]
        )  # min(c[1] for c in bad_children)
        print(
            f"Unidimensional vectors => Accuracy: {accuracy:6.1%} | good: {good:>6,} / {good+bad:>6,} | bad {bad:>6,} / {good+bad:>6,} | bad_children => min_distance: {min_distance:15,.3f}"
        )
        for bad_child in bad_children:
            print(f"    {bad_child}")
    else:
        print(
            f"Unidimensional vectors => Accuracy: {accuracy:6.1%} | good: {good:>6,} / {good+bad:>6,} | bad {bad:>6,} / {good+bad:>6,}"
        )
