import json
import os

import numpy as np
import matplotlib.pyplot as plt

from embeddings import get_embedding


CACHE_FILE = "embeddings_cache.json"


def load_documents(path):

    with open(path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    return documents


def build_embedding_matrix(documents):

    if os.path.exists(CACHE_FILE):

        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            cache = json.load(file)

        if len(cache) == len(documents):
            print("Loaded embeddings from cache.")
            return np.array(cache)

    print("Generating embeddings...")

    embedding_matrix = []

    for document in documents:

        embedding = get_embedding(
            document["text"],
            input_type="passage"
        )

        embedding_matrix.append(embedding)

    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(embedding_matrix, file)

    return np.array(embedding_matrix)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, embedding_matrix, documents, top_k=5):

    query_embedding = get_embedding(
        query,
        input_type="query"
    )

    scores = []

    for i in range(len(embedding_matrix)):

        similarity = cosine_similarity(
            query_embedding,
            embedding_matrix[i]
        )

        scores.append(similarity)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "score": scores[index],
                "topic": documents[index]["topic"],
                "text": documents[index]["text"]
            }
        )

    return results


def pca_via_svd(data, n_components=2):

    mean = np.mean(data, axis=0)
    centered_data = data - mean

    U, S, Vt = np.linalg.svd(centered_data)

    principal_components = Vt[:n_components]

    projected_data = centered_data @ principal_components.T

    return projected_data


def plot_embeddings(embedding_matrix, documents):

    projected_data = pca_via_svd(
        embedding_matrix,
        n_components=2
    )

    topics = []

    for document in documents:
        if document["topic"] not in topics:
            topics.append(document["topic"])

    for topic in topics:

        x = []
        y = []

        for i in range(len(documents)):

            if documents[i]["topic"] == topic:

                x.append(projected_data[i][0])
                y.append(projected_data[i][1])

        plt.scatter(x, y, label=topic)

    plt.title("Semantic Search Embedding Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    documents = load_documents("documents.json")

    embedding_matrix = build_embedding_matrix(documents)

    print("\nSemantic Search")

    while True:

        query = input("\nType in the search or type quit ")

        if query.lower() == "quit":
            break

        results = search(
            query,
            embedding_matrix,
            documents,
            top_k=5
        )

        print()

        for i, result in enumerate(results, start=1):

            print(f"{i}. [{result['topic']}]")
            print(result["text"])
            print(f"Score: {result['score']:.4f}")
            print()

    plot_embeddings(
        embedding_matrix,
        documents
    )