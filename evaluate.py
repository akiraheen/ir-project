import numpy as np
from typing import List, Dict
from PIL import Image
from retriever import CLIPRetrievalSystem
import pandas as pd
import matplotlib.pyplot as plt


class RetrievalEvaluator:
    def __init__(self, retriever: CLIPRetrievalSystem):
        self.retriever = retriever
        self.k_values = [5, 10, 20]  # Different k values for accuracy
        self.max_k = max(self.k_values)  # Maximum k value needed

    def evaluate_transformation(
        self, original_image: Image.Image, transformed_image: Image.Image
    ) -> Dict:
        """
        Evaluate retrieval performance between original and transformed images

        Args:
            original_image: Original image as PIL Image object (baseline)
            transformed_image: Transformed image as PIL Image object
            k: Number of top results to consider for NDCG, MAP, and Recall

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

        ndcg_values = {
            f"ndcg@{k_val}": self._calculate_ndcg(original_ids, transformed_ids, k_val)
            for k_val in self.k_values
        }
        map_values = {
            f"map@{k_val}": self._calculate_map(original_ids, transformed_ids, k_val)
            for k_val in self.k_values
        }
        recall_values = {
            f"recall@{k_val}": self._calculate_recall(
                original_ids, transformed_ids, k_val
            )
            for k_val in self.k_values
        }
        accuracies = {
            f"accuracy@{k_val}": self._calculate_accuracy(
                original_ids, transformed_ids, k_val
            )
            for k_val in self.k_values
        }

        return {
            **ndcg_values,
            **map_values,
            **recall_values,
            **accuracies,  # Include all accuracy metrics
            "original_results": original_results[
                : self.max_k
            ],  # Only return top-k for display
            "transformed_results": transformed_results[: self.max_k],
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


def main():
    retriever = CLIPRetrievalSystem()
    evaluator = RetrievalEvaluator(retriever)

    original_image = Image.open("data/Yummly28K/images27638/img00001.jpg").convert(
        "RGB"
    )
    transformed_image = original_image.rotate(45)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Display transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title("Transformed Image (45° Rotation)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    results = evaluator.evaluate_transformation(
        original_image=original_image, transformed_image=transformed_image
    )

    # Print metrics
    for k in evaluator.k_values:
        print(f"k: {k}")
        print(f"NDCG: {results[f'ndcg@{k}']:.3f}")
        print(f"MAP: {results[f'map@{k}']:.3f}")
        print(f"Recall: {results[f'recall@{k}']:.3f}")
        print(f"Accuracy: {results[f'accuracy@{k}']:.3f}")
        print()

    max_results = len(results["original_results"])

    # Prepare data for the DataFrame
    comparison_data = []
    for i in range(max_results):
        orig_name = (
            results["original_results"][i]["metadata"]["name"]
            if i < len(results["original_results"])
            else ""
        )
        orig_score = (
            results["original_results"][i]["score"]
            if i < len(results["original_results"])
            else float("nan")
        )

        trans_name = (
            results["transformed_results"][i]["metadata"]["name"]
            if i < len(results["transformed_results"])
            else ""
        )
        trans_score = (
            results["transformed_results"][i]["score"]
            if i < len(results["transformed_results"])
            else float("nan")
        )

        comparison_data.append(
            {
                "Rank": i + 1,
                "Original Recipe": orig_name,
                "Original Score": orig_score,
                "Transformed Recipe": trans_name,
                "Transformed Score": trans_score,
            }
        )

    # Create and display the DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Format the scores to 3 decimal places
    pd.set_option("display.precision", 3)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 30)

    print("\nResults Comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
