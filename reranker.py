from retriever import CLIPRetrievalSystem
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.ingredient_filter import clean_ingredients

class Reranker:

    DEFAULT_METADATA_DIR = "data/Yummly28K/metadata27638"
    DEFAULT_IMAGE_DIR = "data/Yummly28K/images27638"
    DEFAULT_INDEX_DIR = "processed_data"

    def __init__(self, retriever: CLIPRetrievalSystem):
        self.retriever = retriever
        self.max_k = 1000

    def bm25_rerank(self, transformed_results):

        # query_id should be the same for all results in transformed_results

        query_id = transformed_results[0]["query_id"]

        query_ingredients = self.retriever.get_ingredients_by_id(query_id)

        # here, docs is the union of all ingredients from the top_k recipes
        docs = [result["metadata"]["ingredientLines"] for result in transformed_results]

        tokenized_docs = []

        for doc in docs:
            lines = clean_ingredients(doc)
            tokens_per_doc = [line.lower().split() for line in lines]
            flat_tokens = [token for line in tokens_per_doc for token in line]
            tokenized_docs.append(flat_tokens)

        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = [
            token for line in clean_ingredients(query_ingredients)
            for token in line.lower().split()
        ]

        scores = bm25.get_scores(tokenized_query)
        sorted_indices = scores.argsort()[::-1]

        reranked_results = []
        for idx in sorted_indices:
            result = transformed_results[idx]
            reranked_results.append({
                # "id": result["metadata"].get("id"),
                "score": result["score"],
                "bm25_score": float(scores[idx]),
                "query_id": result["query_id"],
                "metadata": result["metadata"]
            })

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