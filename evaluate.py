from typing import List, Dict
from PIL import Image

from reranker import Reranker
from retriever import CLIPRetrievalSystem

import matplotlib.pyplot as plt
from transformations import (
    ImageTransformation,
)


class RetrievalEvaluator:
    DATASET_SIZE = 27_638

    def __init__(self, retriever: CLIPRetrievalSystem,
                 reranker: Reranker
                 ):
        self.retriever = retriever
        self.reranker = reranker
        self.max_k = 1000

    def evaluate_transformations(
        self,
        original_image: Image.Image,
        original_id: str,
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
                original_id=original_id, transformed_image=transformed_image
            )
            results[transform.name] = eval_results

        if show_images:
            plt.tight_layout()
            plt.show()

        # Print results for each transformation

        return results

    def evaluate_transformation(
        self, original_id: str, transformed_image: Image.Image, re_ranking: bool = True,
    ) -> Dict:
        """
        Evaluate retrieval performance between original and transformed images

        Args:
            original_id: ID of the original image
            transformed_image: Transformed image as PIL Image object
            re_ranking: reranking is performed if True

        Returns:
            Dictionary containing evaluation metrics
        """
        print("From evaluate, original_id:", original_id)
        file_id = self.retriever.get_number_by_id(original_id)
        print("From evaluate, file_id:", file_id)

        transformed_results = self.retriever.query_with_image_data(
            file_id, transformed_image, top_k=self.max_k
        )

        if re_ranking is True:

            reranked_results = self.reranker.bm25_rerank(transformed_results)
            transformed_ids = [result["metadata"]["id"] for result in reranked_results]

        else:
            transformed_ids = [result["metadata"]["id"] for result in transformed_results]

        # Calculate MRR for each k value
        metrics = {}

        metrics.update({"mrr": self._calculate_mrr(original_id, transformed_ids)})

        return metrics

    def _calculate_mrr(self, original_id, transformed_ids: List):
        """
        Calculate Mean Reciprocal Rank - the multiplicative inverse of the rank
        of the first correct result
        """
        for i, id in enumerate(transformed_ids):
            if id == original_id:
                return 1 / (i + 1)
        return 0.0


def crop_image(image: Image.Image, crop_size: int) -> Image.Image:
    """Crop image to specified size from center"""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))
