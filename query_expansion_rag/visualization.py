import umap
import umap.umap_ as umap
import matplotlib.pyplot


# Retrieve all embeddings from the collection
all_data = chroma_collection.get(include=["embeddings", "metadatas"])
all_embeddings = all_data["embeddings"]

# Fit UMAP on all embeddings
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(all_embeddings)
# umap_transform = umap.UMAP(n_jobs=-1).fit(all_embeddings)
projected_dataset_embeddings = umap_transform.transform(all_embeddings)

# For the retrieved embeddings
retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = umap_transform.transform(original_query_embedding)
projected_augmented_query_embedding = umap_transform.transform(augmented_query_embedding)
projected_retrieved_embeddings = umap_transform.transform(retrieved_embeddings)

plt.figure(figsize=(8, 10))

# Plot entire dataset in gray
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10, 
    color="gray", 
    alpha=0.5, 
    label="All Data Chunks"
)

# Plot retrieved docs in green circles
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="green",
    label="Retrieved Chunks"
)

# Plot original and augmented query points
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="red",
    label="Original Query"
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
    label="Augmented Query"
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"UMAP Projection for Query: {original_query}")
plt.axis("off")
plt.legend()
plt.show()