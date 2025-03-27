# Updated Code: clipretrievalsystem.py

import os
import json
import glob
import random
import numpy as np
import torch
import faiss
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from lambdamart import LambdaRank  # Import LambdaRank

class CLIPRetrievalSystem:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.metadata = []
        self.image_dir = None
        self.index_dir = "processed_data"  # Directory to store processed files

        # Create directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        # Initialize LambdaRank model for re-ranking
        self.lambdarank = LambdaRank()

    def process_and_save_dataset(self, metadata_dir, image_dir, force_reprocess=False):
        """Process dataset and save features or load existing"""
        self.image_dir = image_dir

        # Define paths for saved data
        index_path = os.path.join(self.index_dir, "dataset.index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        config_path = os.path.join(self.index_dir, "config.json")

        if not force_reprocess and self._check_existing_data(index_path, metadata_path, config_path):
            print("Loading preprocessed data...")
            self._load_processed_data(index_path, metadata_path, config_path)
            return

        print("Processing dataset from scratch...")
        self._process_dataset(metadata_dir, image_dir)
        self._save_processed_data(index_path, metadata_path, config_path)

    def _check_existing_data(self, index_path, metadata_path, config_path):
        """Check if all required files exist"""
        return all(os.path.exists(p) for p in [index_path, metadata_path, config_path])

    def _save_processed_data(self, index_path, metadata_path, config_path):
        """Save all processed data with proper error handling"""
        if self.index is None:
            raise RuntimeError("Cannot save data: FAISS index not initialized")
        if not self.metadata:
            raise RuntimeError("Cannot save data: Metadata is empty")

        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            print(f"Saved index with {self.index.ntotal} embeddings")

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            print(f"Saved metadata for {len(self.metadata)} recipes")

            # Save config
            with open(config_path, 'w') as f:
                json.dump({
                    "image_dir": self.image_dir,
                    "embedding_dim": self.index.d
                }, f)
            print("Saved configuration file")

        except Exception as e:
            raise RuntimeError(f"Failed to save processed data: {str(e)}")

    def _load_processed_data(self, index_path, metadata_path, config_path):
        """Load previously processed data"""
        # Load index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.image_dir = config["image_dir"]

        print(f"Loaded {len(self.metadata)} recipes with {self.index.ntotal} embeddings")

    def _process_dataset(self, metadata_dir, image_dir):
        """Modified with better error handling"""
        metadata_files = glob.glob(os.path.join(metadata_dir, "meta*.json"))
        print(f"Found {len(metadata_files)} metadata files")  # Debug

        text_embeddings = []
        image_embeddings = []
        self.metadata = []

        for meta_file in tqdm(metadata_files, desc="Processing dataset"):
            try:
                base_name = os.path.basename(meta_file)
                file_id = base_name[4:-5]
                img_file = os.path.join(image_dir, f"img{file_id}.jpg")

                if not os.path.exists(img_file):
                    print(f"Missing image: {img_file}")  # Debug
                    continue

                with open(meta_file, 'r') as f:
                    recipe = json.load(f)

                # Text processing
                text = self._format_recipe_text(recipe)
                text_inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                # Image processing
                image = Image.open(img_file).convert("RGB")
                image_inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    text_embed = self.model.get_text_features(**text_inputs).cpu().numpy()
                    image_embed = self.model.get_image_features(**image_inputs).cpu().numpy()

                text_embeddings.append(text_embed)
                image_embeddings.append(image_embed)
                self.metadata.append(recipe)

            except Exception as e:
                print(f"Error processing {meta_file}: {str(e)}")
                continue

        # Validate embeddings
        if not text_embeddings or not image_embeddings:
            raise ValueError("No valid embeddings generated. Check your data paths and files.")

        try:
            combined_embeddings = np.concatenate([
                np.vstack(text_embeddings),
                np.vstack(image_embeddings)
            ], axis=0)

            self.index = faiss.IndexFlatIP(combined_embeddings.shape[1])
            self.index.add(combined_embeddings.astype('float32'))
            print(f"Created index with {self.index.ntotal} embeddings")

        except Exception as e:
            raise RuntimeError(f"Index creation failed: {str(e)}")

    def _format_recipe_text(self, recipe):
        """Format recipe metadata using actual JSON fields from Yummly28K"""
        components = [
            f"Title: {recipe.get('name', 'Unknown')}",
            f"Ingredients: {', '.join(recipe.get('ingredientLines', []))}",
            f"Cuisine: {recipe.get('cuisine', 'Unknown')}",
            f"Course: {recipe.get('course', 'Unknown')}",
            f"Flavors: {recipe.get('flavors', 'Unknown')}",
            f"Total Time: {recipe.get('totalTime', 'Unknown')}",
            f"Nutrition: {', '.join([str(n) for n in recipe.get('nutritionEstimates', [])])}"
        ]
        return ". ".join(components)

    # In clipretrievalsystem.py

    def query_with_image(self, image_path, top_k=5, exclude_self=False):
        """Query the system with an image"""
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            raise ValueError("Invalid image path")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            query_embed = self.model.get_image_features(**inputs).cpu().numpy()

        distances, indices = self.index.search(query_embed.astype('float32'), top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            original_idx = idx % len(self.metadata)

            # Skip exact match
            if exclude_self and os.path.basename(image_path) == f"img{self.metadata[original_idx]['id']}.jpg":
                continue

            results.append({
                "score": float(score),
                "metadata": self.metadata[original_idx]
            })

            if len(results) >= top_k:
                break

        return results

    def re_rank_with_lambdamart(self, query_results):
        """Re-rank the results using LambdaMART"""
        feature_vectors = []
        relevance_labels = []

        for idx, result in enumerate(query_results):
            score = result['score']  # FAISS similarity score
            relevance = max(1, min(int(score / 10), 10))  # Example: Mapping score to range [1-10]

            # Extracting embeddings for image and text
            image_embed = result.get('image_embed', np.zeros(512))
            text_embed = result.get('text_embed', np.zeros(512))

            # Construct feature vector (including score and embeddings)
            feature_vector = np.concatenate([np.array([score]), image_embed.flatten(), text_embed.flatten()])

            feature_vectors.append(feature_vector)
            relevance_labels.append(relevance)


        feature_vectors = np.array(feature_vectors)
        relevance_labels = np.array(relevance_labels)


        scaler = StandardScaler()
        feature_vectors = scaler.fit_transform(feature_vectors)


        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(-1, 1)


        self.lambdarank.fit(feature_vectors, relevance_labels)


        reranked_scores = self.lambdarank.predict(feature_vectors)

        # Add the reranked scores to the results and sort by the new scores
        for i, result in enumerate(query_results):
            result['rerank_score'] = reranked_scores[i]

        # Sort results by the new re-ranked scores
        reranked_results = sorted(query_results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked_results


if __name__ == "__main__":
    retriever = CLIPRetrievalSystem()


    retriever.process_and_save_dataset(
        metadata_dir="/Users/shivangikachole/Downloads/Yummly28K/metadata27638",
        image_dir="/Users/shivangikachole/Downloads/Yummly28K/images27638",
        force_reprocess=False
    )

    # Query the system with an image (initial query without re-ranking)
    image_path = "/Users/shivangikachole/Downloads/Yummly28K/images27638/img00005.jpg"
    results = retriever.query_with_image(image_path)
    print("Top result:", results[0]['metadata']['name'])

    # Re-rank the results using LambdaMART
    reranked_results = retriever.re_rank_with_lambdamart(results)

    # Print the re-ranked results
    print("\nTop results after re-ranking:")
    for idx, result in enumerate(reranked_results):
        print(f"{idx + 1}. {result['metadata']['name']} (Re-ranked Score: {result['rerank_score']})")

