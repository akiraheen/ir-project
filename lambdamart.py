# lambdamart.py (LambdaRank class update)

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class LambdaRank:
    def __init__(self):
        # Initialize GradientBoostingRegressor for LambdaRank
        self.model = GradientBoostingRegressor()

    def fit(self, feature_vectors, labels):
        """
        Train the LambdaRank model using feature vectors and corresponding relevance labels.
        Args:
            feature_vectors: List or np.array of feature vectors (e.g., image-text embeddings)
            labels: List or np.array of relevance scores (e.g., relevance labels for query-result pairs)
        """
        # Fit the model with feature vectors and labels
        self.model.fit(feature_vectors, labels)

    def rerank_results(self, results):
        """
        Re-rank the search results using the trained LambdaRank model.
        Args:
            results: List of results with features to re-rank (e.g., distance scores)
        Returns:
            List of re-ranked results with updated scores.
        """
        feature_vectors = []
        for result in results:
            feature_vectors.append(result["features"])  # Assuming "features" is part of each result

        # Make predictions (re-rank the results)
        predicted_scores = self.predict(feature_vectors)

        # Add predicted scores to the results
        for idx, result in enumerate(results):
            result["rerank_score"] = predicted_scores[idx]

        return results

    def predict(self, X):
        """Predict the scores using the trained model"""
        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model has not been trained yet")
        return self.model.predict(X)
