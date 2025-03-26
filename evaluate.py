import math
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
from retriever import CLIPRetrievalSystem
import pandas as pd
import matplotlib.pyplot as plt


class RetrievalEvaluator:
    def __init__(self, retriever: CLIPRetrievalSystem):
        self.retriever = retriever
        self.k_values = [5, 10, 20]
        self.max_k = max(self.k_values)

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


def crop_image(image: Image.Image, crop_size: int) -> Image.Image:
    """Crop image to specified size from center"""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def run_evaluation(
    evaluator: RetrievalEvaluator,
    original_image: Image.Image,
    transformed_image: Image.Image,
    title: str,
) -> None:
    """Run evaluation and display results for a pair of images"""
    # Display images side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image ({title})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title(f"Transformed Image ({title})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Run evaluation
    results = evaluator.evaluate_transformation(
        original_image=original_image, transformed_image=transformed_image
    )

    # Print metrics
    print(f"\nResults for {title}:")
    print("-" * 50)
    for k in evaluator.k_values:
        print(f"\nMetrics at k={k}:")
        print(f"NDCG: {results[f'ndcg@{k}']:.3f}")
        print(f"MAP: {results[f'map@{k}']:.3f}")
        print(f"Recall: {results[f'recall@{k}']:.3f}")
        print(f"Accuracy: {results[f'accuracy@{k}']:.3f}")

    comparison_data = []
    for i in range(evaluator.k_values[0]):  # Use smallest k for display
        orig_result = results["original_results"][i]
        trans_result = results["transformed_results"][i]

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
    print(f"\nTop {evaluator.k_values[0]} Results Comparison:")
    print(comparison_df.to_string(index=False))


def main():
    retriever = CLIPRetrievalSystem()
    evaluator = RetrievalEvaluator(retriever)
    original_image = Image.open("data/Yummly28K/images27638/img00001.jpg").convert(
        "RGB"
    )

    # 1. Evaluate with normal rotation
    normal_transformed = original_image.rotate(45)
    run_evaluation(evaluator, original_image, normal_transformed, "Rotated")

    # 2. Evaluate with cropped rotation
    crop_size = min(original_image.size)
    diagonal = int(crop_size / math.sqrt(2))

    cropped_original = crop_image(original_image, diagonal)
    cropped_transformed = crop_image(original_image.rotate(45), diagonal)

    run_evaluation(
        evaluator,
        cropped_original,
        cropped_transformed,
        "Rotated + Cropped",
    )


if __name__ == "__main__":
    main()
