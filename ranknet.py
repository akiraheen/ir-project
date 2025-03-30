import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PIL import Image

class RankNet(nn.Module):
    def __init__(self, embedding_dim):
        super(RankNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single ranking score
        )

    def forward(self, x):
        return self.fc(x)

def generate_training_pairs(retriever, num_samples=10000):
    """
    Generate training pairs from FAISS retrieval results.
    """
    training_pairs = []
    for _ in range(num_samples):
        idx = random.randint(0, len(retriever.metadata) - 1)
        query_embedding = retriever.index.reconstruct(idx).astype(np.float32)

        # Retrieve top-K similar items
        _, indices = retriever.index.search(query_embedding.reshape(1, -1), k=10)
        retrieved_indices = indices[0]

        # Generate positive and negative pairs
        for i in range(len(retrieved_indices) - 1):
            s1_idx, s2_idx = int(retrieved_indices[i]), int(retrieved_indices[i + 1])

            s1_embed = retriever.index.reconstruct(s1_idx) if retriever.index.is_trained else retriever.index.xb[s1_idx]
            s2_embed = retriever.index.reconstruct(s2_idx) if retriever.index.is_trained else retriever.index.xb[s2_idx]

            # Generate pairwise preference labels
            training_pairs.append((s1_embed, s2_embed, 1))  # s1 > s2
            training_pairs.append((s2_embed, s1_embed, 0))  # s2 > s1

    return training_pairs

def train_ranknet(ranknet, retriever, epochs=5, batch_size=32, lr=0.001):
    """
    Train RankNet using pairwise ranking loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranknet.to(device)
    optimizer = optim.Adam(ranknet.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    training_data = generate_training_pairs(retriever, num_samples=10000)
    random.shuffle(training_data)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            s1_embeds, s2_embeds, labels = zip(*batch)
            s1_embeds = torch.tensor(np.array(s1_embeds), dtype=torch.float32).to(device)
            s2_embeds = torch.tensor(np.array(s2_embeds), dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            s1_scores = ranknet(s1_embeds).squeeze()
            s2_scores = ranknet(s2_embeds).squeeze()

            # Apply sigmoid transformation
            loss = criterion(torch.sigmoid(s1_scores - s2_scores), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    print("Training complete!")

def rerank_results(ranknet, retriever, query_image_path, top_k=5):
    """
    Rerank FAISS-retrieved results using RankNet and print the top results with scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranknet.to(device)
    ranknet.eval()

    results = retriever.query_with_image(query_image_path, top_k=top_k)

    print("\nOriginal FAISS Ranking:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['metadata']['name']} - FAISS Score: {result['score']:.4f}")


    query_image = Image.open(query_image_path).convert("RGB")
    inputs = retriever.processor(images=query_image, return_tensors="pt").to(device)

    with torch.no_grad():
        query_embed = retriever.model.get_image_features(**inputs).cpu().numpy()

    reranked_results = []
    for result in results:
        image_id = result["metadata"]["id"]
        image_idx = next(i for i, m in enumerate(retriever.metadata) if m["id"] == image_id)

        image_embed = retriever.index.reconstruct(image_idx)
        image_embed_tensor = torch.tensor(image_embed, dtype=torch.float32).to(device)

        with torch.no_grad():
            score = ranknet(image_embed_tensor.unsqueeze(0)).item()

        reranked_results.append((score, result))

    reranked_results.sort(reverse=True, key=lambda x: x[0])

    print("\nReranked Results:")
    for i, (score, result) in enumerate(reranked_results):
        print(f"{i+1}. {result['metadata']['name']} - RankNet Score: {score:.4f}")

    return [r[1] for r in reranked_results]


