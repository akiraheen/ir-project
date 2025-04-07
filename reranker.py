from typing import Literal
from retriever import CLIPRetrievalSystem
from rank_bm25 import BM25Okapi
from utils.ingredient_filter import clean_ingredients

RerankerName = Literal["jaccard", "bm25"]


class Reranker:
    DEFAULT_METADATA_DIR = "data/Yummly28K/metadata27638"
    DEFAULT_IMAGE_DIR = "data/Yummly28K/images27638"
    DEFAULT_INDEX_DIR = "processed_data"

    def __init__(self, retriever: CLIPRetrievalSystem):
        self.retriever = retriever
        self.max_k = 1000

    def tokenize_ingredients(self, ingredient_lines):
        lines = clean_ingredients(ingredient_lines)
        tokens = [token for line in lines for token in line.lower().split()]
        return tokens

    def preprocess_for_reranking(self, transformed_results):
        query_id = transformed_results[0]["query_id"]
        query_ingredients = self.retriever.get_ingredients_by_id(query_id)
        query_tokens = self.tokenize_ingredients(query_ingredients)

        doc_tokens_list = []
        for result in transformed_results:
            doc_ingredients = result["metadata"]["ingredientLines"]
            tokens = self.tokenize_ingredients(doc_ingredients)
            doc_tokens_list.append(tokens)

        return query_tokens, doc_tokens_list

    def bm25_rerank(self, transformed_results):
        query_tokens, doc_tokens_list = self.preprocess_for_reranking(
            transformed_results
        )
        bm25 = BM25Okapi(doc_tokens_list)

        scores = bm25.get_scores(query_tokens)
        sorted_indices = scores.argsort()[::-1]

        reranked_results = []
        for idx in sorted_indices:
            result = transformed_results[idx]
            reranked_results.append(
                {
                    "score": result["score"],
                    "bm25_score": float(scores[idx]),
                    "query_id": result["query_id"],
                    "metadata": result["metadata"],
                }
            )

        return reranked_results

    def jaccard_rerank(self, transformed_results):
        query_tokens, doc_tokens_list = self.preprocess_for_reranking(
            transformed_results
        )
        query_set = set(query_tokens)

        jaccard_scores = []
        for doc_tokens in doc_tokens_list:
            doc_set = set(doc_tokens)
            intersection = query_set & doc_set
            union = query_set | doc_set
            score = len(intersection) / len(union) if union else 0.0
            jaccard_scores.append(score)

        sorted_indices = sorted(
            range(len(jaccard_scores)), key=lambda i: jaccard_scores[i], reverse=True
        )

        reranked_results = []
        for idx in sorted_indices:
            result = transformed_results[idx]
            reranked_results.append(
                {
                    "score": result["score"],
                    "jaccard_score": float(jaccard_scores[idx]),
                    "query_id": result["query_id"],
                    "metadata": result["metadata"],
                }
            )

        return reranked_results

    # def jaccard_similarity(self, text1, text2):
    #
    #     set1 = set(text1.lower().split())  # Convert to lowercase and split into words
    #     set2 = set(text2.lower().split())
    #
    #     intersection = len(set1 & set2)
    #     union = len(set1 | set2)
    #
    #     return intersection / union if union != 0 else 0
    #
    # def tfidf_sim(self, text1, text2):
    #     vectorizer = TfidfVectorizer()
    #
    #     # Fit and transform both texts as whole documents
    #     tfidf_matrix = vectorizer.fit_transform([text1, text2])
    #
    #     # Compute cosine similarity
    #     cs = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
    #     return cs
    #
    # def bert_sim(self, text1, text2):
    #
    #     model = SentenceTransformer("all-MiniLM-L6-v2")
    #
    #     emb1 = model.encode(text1)
    #     emb2 = model.encode(text2)
    #
    #     # Compute cosine similarity
    #     cs = cosine_similarity([emb1], [emb2])[0, 0]
    #     return cs
