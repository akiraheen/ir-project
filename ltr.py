from sklearn.model_selection import train_test_split
import lightgbm as lightgbm

class LTR:
    def __init__(self):
        pass

    def lambdaMart(self, dataframe):
        unique_qids = dataframe["qid"].unique()

        qid_train, qid_test = train_test_split(unique_qids, test_size=0.2, random_state=42)

        # split based on qid
        train = dataframe[dataframe["qid"].isin(qid_train)]
        test = dataframe[dataframe["qid"].isin(qid_test)]

        print(f"No of unique queries in train : {train['qid'].nunique()}")
        print(f"No of unique queries in test : {test['qid'].nunique()}")

        qids_train = train.groupby("qid")["qid"].count().to_numpy()
        X_train = train.drop(["qid", "label", "relevant_docId", "ingredients", "name",
                              "relevant_name", "relevant_ingredients"], axis=1)
        y_train = train["label"]

        qids_test = test.groupby("qid")["qid"].count().to_numpy()
        X_test = test.drop(["qid", "label", "relevant_docId", "ingredients", "name",
                            "relevant_name", "relevant_ingredients"], axis=1)
        y_test = test["label"]

        ranker = lightgbm.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=5,
            importance_type="gain",
            metric="ndcg",
            num_leaves=10,
            learning_rate=0.05,
            max_depth=-1,
            label_gain=[i for i in range(max(y_train.max(), y_test.max()) + 1)],
            min_gain_to_split=0.0)

        evals_result = {}

        ranker.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[ (X_test, y_test)],
            eval_group=[qids_test],
            eval_at=[3,5],
            eval_metric="ndcg",
            callbacks=[lightgbm.record_evaluation(evals_result)])

        # print("\nEvaluation Results:")
        # for eval_set, scores in evals_result.items():
        #     print(f"\nMetrics for {eval_set}:")
        #     for metric, values in scores.items():
        #         print(f"{metric}: {values[-1]:.4f} (last iteration)")

        return ranker

    def ltr_with_image_data(self, new_query_df, ranker):
        """
           Given a new query and candidate images, rerank them using BM25

           Parameters:
           - new_query_df (DataFrame): A DataFrame containing features of candidate images for the query.
           - ranker (LGBMRanker): The trained LambdaMART model.

           Returns:
           - reranked_df (DataFrame): A sorted DataFrame of images based on ranking scores.
           """

        # Ensure the same feature columns as training (remove non-feature columns)
        X_new_query = new_query_df.drop(["qid", "relevant_docId", "ingredients", "name",
                                         "relevant_name", "relevant_ingredients", "jaccard_score"], axis=1)

        # Predict ranking scores
        new_query_df["predicted_score"] = ranker.predict(X_new_query)

        # Sort results based on predicted score (descending)
        reranked_df = new_query_df.sort_values(by="predicted_score", ascending=False)

        return reranked_df[["qid", "relevant_docId", "predicted_score"]]