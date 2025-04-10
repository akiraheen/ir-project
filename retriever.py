import os
import json
import glob
from typing import Optional
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# import optuna
# import optuna.integration.lightgbm as optuna_lgb
import time
# from collections import OrderedDict
# from sklearn.model_selection import ParameterGrid
# from nltk.tokenize import word_tokenize


class CLIPRetrievalSystem:
    DEFAULT_METADATA_DIR = "data/Yummly28K/metadata27638"
    DEFAULT_IMAGE_DIR = "data/Yummly28K/images27638"
    DEFAULT_INDEX_DIR = "processed_data"

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        metadata_dir=DEFAULT_METADATA_DIR,
        image_dir=DEFAULT_IMAGE_DIR,
        index_dir=DEFAULT_INDEX_DIR,
        process_dataset: bool = True,
        force_reprocess: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)  # type: ignore
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.index = None
        self.metadata = []
        self.image_dir = None
        self.index_dir = index_dir
        self.metadata_dir = metadata_dir
        self.image_dir = image_dir

        # Create directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        if process_dataset:
            self.process_and_save_dataset(force_reprocess=force_reprocess)

    def get_metadata(self, image_id: str):
        """Get metadata for an image by ID"""
        for recipe in self.metadata:
            if recipe["id"] == image_id:
                return recipe
        return None

    def process_and_save_dataset(
        self,
        metadata_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        force_reprocess: bool = False,
    ):
        """Process dataset and save features or load existing"""
        metadata_dir = metadata_dir or self.metadata_dir or self.DEFAULT_METADATA_DIR
        image_dir = image_dir or self.image_dir or self.DEFAULT_IMAGE_DIR

        # Define paths for saved data
        index_path = os.path.join(self.index_dir, "dataset.index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        config_path = os.path.join(self.index_dir, "config.json")

        if not force_reprocess and self._check_existing_data(
            index_path, metadata_path, config_path
        ):
            print("Loading preprocessed data...")
            self._load_processed_data(
                index_path, metadata_path, config_path=config_path
            )
            return

        print("Processing dataset from scratch...")
        self._process_dataset(self.metadata_dir, self.image_dir)
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
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
            print(f"Saved metadata for {len(self.metadata)} recipes")

            # Save config
            with open(config_path, "w") as f:
                json.dump(
                    {"image_dir": self.image_dir, "embedding_dim": self.index.d}, f
                )
            print("Saved configuration file")

        except Exception as e:
            raise RuntimeError(f"Failed to save processed data: {str(e)}")

    def _load_processed_data(self, index_path, metadata_path, config_path):
        """Load previously processed data"""
        # Load index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
            self.image_dir = config["image_dir"]

        print(
            f"Loaded {len(self.metadata)} recipes with {self.index.ntotal} embeddings"
        )

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

                with open(meta_file, "r") as f:
                    recipe = json.load(f)

                recipe["file_id"] = file_id

                # Text processing
                text = self._format_recipe_text(recipe)
                text_inputs = self.processor(  # type: ignore
                    text=text, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # Image processing
                image = Image.open(img_file).convert("RGB")
                image_inputs = self.processor(images=image, return_tensors="pt").to(  # type: ignore
                    self.device
                )

                with torch.no_grad():
                    text_embed = (
                        self.model.get_text_features(**text_inputs).cpu().numpy()  # type: ignore
                    )
                    image_embed = (
                        self.model.get_image_features(**image_inputs).cpu().numpy()  # type: ignore
                    )

                text_embeddings.append(text_embed)
                image_embeddings.append(image_embed)
                self.metadata.append(recipe)

            except Exception as e:
                print(f"Error processing {meta_file}: {str(e)}")
                continue

        # Validate embeddings
        if not text_embeddings or not image_embeddings:
            raise ValueError(
                "No valid embeddings generated. Check your data paths and files."
            )

        try:
            combined_embeddings = np.concatenate(
                [np.vstack(text_embeddings), np.vstack(image_embeddings)], axis=0
            )

            self.index = faiss.IndexFlatIP(combined_embeddings.shape[1])
            self.index.add(combined_embeddings.astype("float32"))
            # self._generate_relevant_recipes()
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
            f"Nutrition: {', '.join([str(n) for n in recipe.get('nutritionEstimates', [])])}",
        ]
        return ". ".join(components)

    def query_with_image(self, image_path: str, top_k=5, exclude_self=False):
        """Query the system with an image path"""
        try:
            image = Image.open(image_path).convert("RGB")
            exclude_path = image_path if exclude_self else None
            return self.query_with_image_data(image, top_k, exclude_path)
        except Exception as e:
            raise ValueError(f"Invalid image path: {str(e)}")

    def query_with_image_data(
        self, file_id, image: Image.Image, top_k=5, exclude_path: Optional[str] = None
    ):
        """Query the system with a PIL Image object"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # type: ignore

        with torch.no_grad():
            query_embed = self.model.get_image_features(**inputs).cpu().numpy()  # type: ignore

        distances, indices = self.index.search(query_embed.astype("float32"), top_k)  # type: ignore

        results = []
        for idx, score in zip(indices[0], distances[0]):
            original_idx = idx % len(self.metadata)
            img_name = f"img{self.metadata[original_idx]['id']}.jpg"
            # print(img_name)
            if exclude_path and os.path.basename(exclude_path) == img_name:
                print(f"Skipping {exclude_path} because it's the query image")
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[original_idx],
                    "query_id": file_id,
                }
            )

            if len(results) >= top_k:
                break

        return results

    def get_number_by_id(self, id):
        for item in self.metadata:
            if item.get("id") == id:
                return item.get("file_id")
        return None

    def get_ingredients_by_id(self, recipe_id):
        for item in self.metadata:
            if item.get("file_id") == recipe_id:
                return item.get("ingredientLines")
        return None


if __name__ == "__main__":
    start_time = time.time()
    # retriever = CLIPRetrievalSystem()

    retriever = CLIPRetrievalSystem(
        metadata_dir="data/Yummly28K/metadata27638",
        image_dir="data/Yummly28K/images27638",
    )

    # Test query
    results = retriever.query_with_image("data/Yummly28K/images27638/img00001.jpg")
    print("Top result:", results[0]["metadata"]["name"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
