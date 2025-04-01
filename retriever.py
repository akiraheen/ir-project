import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from utils.ingredient_filter import extract_ingredients
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

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
                # json.dump(self.metadata, f)
                json.dump(self.metadata, f, indent=4)
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

                recipe["qid"] = file_id

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
            self._generate_relevant_recipes()
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

    def _generate_relevant_recipes(self, top_k=5):
        """Generates relevant recipes per query with FAISS similarity, cosine similarity, and relevance labels."""
        print("Generating relevant recipes per query...")

        for i, query_recipe in enumerate(self.metadata):
            # Retrieve FAISS embedding of query recipe
            query_embed = self.index.reconstruct(i).reshape(1, -1)

            # Perform FAISS similarity search
            distances, indices = self.index.search(query_embed.astype('float32'), top_k + 1)

            relevant_recipes = []

            for idx, faiss_score in zip(indices[0], distances[0]):
                if idx == i:
                    continue  # Skip exact match

                # Retrieve the embedding of the retrieved recipe
                retrieved_embed = self.index.reconstruct(int(idx)).reshape(1, -1)

                # Compute cosine similarity
                cosine_sim = cosine_similarity(query_embed, retrieved_embed)[0][0]

                # Assign relevance labels (0-3 scale)
                label = pd.cut(
                    [cosine_sim], bins=[-1, 0.5, 0.7, 0.9, 1], labels=[0, 1, 2, 3], include_lowest=True
                ).astype(int)[0]  # Extract the single value from the series

                # Store FAISS score, cosine similarity, and relevance label
                relevant_recipes.append({
                    "qid": self.metadata[idx]['qid'],
                    "name": self.metadata[idx]['name'],
                    "faiss_score": float(faiss_score),
                    "cosine_sim": float(cosine_sim),
                    "label": int(label),
                    "ingredients": extract_ingredients(list(self.metadata[idx]['ingredientLines']))
                })

                if len(relevant_recipes) >= top_k:
                    break

            query_recipe["relevant_recipes"] = relevant_recipes

def create_dataframe(metadata_path):

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    recipe_data = []
    for recipe in metadata:
        qid = recipe["qid"]
        ingredients = " ".join(extract_ingredients(recipe["ingredientLines"]))
        name = recipe["name"]

        for r in recipe.get("relevant_recipes", []):
            relevant_ingredients = " ".join(r.get("ingredients", []))
            jaccard_score = jaccard_similarity(ingredients, relevant_ingredients)
            ##bm25
            ##other weighting techniques

            recipe_data.append({
                "qid": qid,
                "ingredients": ingredients,
                "name": name,

                "relevant_docId": r["qid"],
                "relevant_name": r["name"],
                "relevant_ingredients": relevant_ingredients,
                "jaccard_similarity": jaccard_score
            })

    return pd.DataFrame(recipe_data)


def jaccard_similarity(text1, text2):

    set1 = set(text1.lower().split())  # Convert to lowercase and split into words
    set2 = set(text2.lower().split())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union != 0 else 0

if __name__ == "__main__":
    retriever = CLIPRetrievalSystem()

    # Fixed path typo: Yumm28K → Yummly28K
    retriever.process_and_save_dataset(
        metadata_dir="../Yummly28K/metadata20",  # Change to full dataset
        image_dir="../Yummly28K/images20",
        force_reprocess=False  # Run first time to generate files
    )

    # Test query
    results = retriever.query_with_image("../Yummly28K/images27638/img00001.jpg")
    print("Top result:", results[0]['metadata']['name'])

    df = create_dataframe("processed_data/metadata.json")

    df["label"] = pd.qcut(df["jaccard_similarity"], q=5, labels=[0, 1, 2, 3], duplicates="drop")
    df["label"] = df["label"].astype(int)  # Convert to integer

    df.to_csv("out.csv", index=False)
    df.to_json("out.json", indent=5)