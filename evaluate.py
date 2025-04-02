from pathlib import Path
import numpy as np

from typing import List, Dict
from PIL import Image

from retriever import CLIPRetrievalSystem
import pandas as pd
import matplotlib.pyplot as plt
from transformations import (
    ImageTransformation,
)


class RetrievalEvaluator:
    def __init__(self, retriever: CLIPRetrievalSystem):
        self.retriever = retriever
        self.k_values = [5, 10, 20]
        self.max_k = max(self.k_values)

    def evaluate_transformations(
        self,
        original_image: Image.Image,
        transformations: List[ImageTransformation],
        show_images: bool = True,
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple transformations against the original image

        Args:
            original_image: The original image to transform
            transformations: List of transformation objects to apply

        Returns:
            Dictionary mapping transformation names to their evaluation results
        """
        results = {}

        n_transforms = len(transformations)

        if show_images:
            # Create subplot grid
            plt.figure(figsize=(15, 5 * (n_transforms + 1)))

            # Show original image
            plt.subplot(n_transforms + 1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

        # Evaluate each transformation
        for idx, transform in enumerate(transformations, 1):
            # Apply transformation
            transformed_image = transform(original_image)

            if show_images:
                # Show transformed image
                plt.subplot(n_transforms + 1, 2, idx * 2 + 1)
                plt.imshow(transformed_image)
                plt.title(f"Transformed Image ({transform.name})")
                plt.axis("off")

            # Evaluate transformation
            eval_results = self.evaluate_transformation(
                original_image, transformed_image
            )
            results[transform.name] = eval_results

        if show_images:
            plt.tight_layout()
            plt.show()

        # Print results for each transformation

        return results

    def evaluate_transformation(
        self, original_image: Image.Image, transformed_image: Image.Image
    ) -> Dict:
        """
        Evaluate retrieval performance between original and transformed images

        Args:
            original_image: Original image as PIL Image object (baseline)
            transformed_image: Transformed image as PIL Image object

        Returns:
            Dictionary containing evaluation metrics
        """
        # Get more results for accuracy calculation
        original_results = self.retriever.query_with_image_data(
            original_image, top_k=self.max_k
        )
        transformed_results = self.retriever.query_with_image_data(
            transformed_image, top_k=self.max_k
        )

        original_ids = [result["metadata"]["id"] for result in original_results]
        transformed_ids = [result["metadata"]["id"] for result in transformed_results]

        # Calculate metrics for each k value
        metrics = {}
        for k in self.k_values:
            metrics.update(
                {
                    f"ndcg@{k}": self._calculate_ndcg(original_ids, transformed_ids, k),
                    f"map@{k}": self._calculate_map(original_ids, transformed_ids, k),
                    f"recall@{k}": self._calculate_recall(
                        original_ids, transformed_ids, k
                    ),
                    f"accuracy@{k}": self._calculate_accuracy(
                        original_ids, transformed_ids, k
                    ),
                }
            )

        return {
            **metrics,
            "original_results": original_results,
            "transformed_results": transformed_results,
        }

    def _calculate_accuracy(
        self, original_ids: List, transformed_ids: List, k: int
    ) -> float:
        """
        Calculate accuracy as the proportion of exact matches in the top-k results
        regardless of their order
        """
        original_set = set(original_ids[:k])
        transformed_set = set(transformed_ids[:k])
        return len(original_set & transformed_set) / k if k > 0 else 0.0

    def _calculate_ndcg(
        self, original_ids: List, transformed_ids: List, k: int
    ) -> float:
        """Calculate NDCG@k"""
        relevance = [1.0 if id in original_ids else 0.0 for id in transformed_ids]
        ideal_relevance = sorted(relevance, reverse=True)

        dcg = self._dcg(relevance[:k])
        idcg = self._dcg(ideal_relevance[:k])

        return dcg / idcg if idcg > 0 else 0.0

    def _dcg(self, relevance: List[float]) -> float:
        """Calculate Discounted Cumulative Gain"""
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    def _calculate_map(
        self, original_ids: List, transformed_ids: List, k: int
    ) -> float:
        """Calculate MAP@k"""
        ap = 0.0
        relevant_count = 0

        for i, id in enumerate(transformed_ids[:k], 1):
            if id in original_ids:
                relevant_count += 1
                precision_at_i = relevant_count / i
                ap += precision_at_i

        return ap / min(k, len(original_ids)) if original_ids else 0.0

    def _calculate_recall(
        self, original_ids: List, transformed_ids: List, k: int
    ) -> float:
        """Calculate Recall@k"""
        relevant_retrieved = len(set(original_ids) & set(transformed_ids[:k]))
        return relevant_retrieved / len(original_ids) if original_ids else 0.0

    def write_comparison_results(
        self, results: Dict[str, Dict], file_path: Path
    ) -> None:
        """Write comparative results for all transformations to a file"""
        with open(file_path, "w") as f:
            # Write metrics for each transformation
            for transform_name, transform_results in results.items():
                f.write(f"\nResults for {transform_name}:\n")
                f.write("-" * 50 + "\n")
                for k in self.k_values:
                    f.write(f"\nMetrics at k={k}:\n")
                    f.write(f"NDCG: {transform_results[f'ndcg@{k}']:.3f}\n")
                    f.write(f"MAP: {transform_results[f'map@{k}']:.3f}\n")
                    f.write(f"Recall: {transform_results[f'recall@{k}']:.3f}\n")
                    f.write(f"Accuracy: {transform_results[f'accuracy@{k}']:.3f}\n")

                # Write top results comparison
                comparison_data = []
                for i in range(self.k_values[0]):
                    orig_result = transform_results["original_results"][i]
                    trans_result = transform_results["transformed_results"][i]

                    comparison_data.append(
                        {
                            "Rank": i + 1,
                            "Original Recipe": orig_result["metadata"]["name"],
                            "Original Score": f"{orig_result['score']:.3f}",
                            "Transformed Recipe": trans_result["metadata"]["name"],
                            "Transformed Score": f"{trans_result['score']:.3f}",
                        }
                    )

                comparison_df = pd.DataFrame(comparison_data)
                f.write(f"\nTop {self.k_values[0]} Results Comparison:\n")
                f.write(comparison_df.to_string(index=False) + "\n")


def crop_image(image: Image.Image, crop_size: int) -> Image.Image:
    """Crop image to specified size from center"""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))
