import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# Load merged OCR + caption data
merged_df = pd.read_csv("merged_results.csv")
texts = (merged_df["Text"].fillna("") + "\n\n" + merged_df["caption"].fillna("")).tolist()
pages = merged_df["Page"].tolist()

# Load queries
query_df = pd.read_csv("HW2_query.csv")
queries = query_df["Question"].tolist()

# Load models
embedding_model = SentenceTransformer("intfloat/e5-large-v2").to("cuda")
query_embeddings = embedding_model.encode(queries, convert_to_tensor=True)
text_embeddings = embedding_model.encode(texts, convert_to_tensor=True)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Retrieval and Re-ranking
top_k = 40
pred = []

for query, query_embedding in tqdm(zip(queries, query_embeddings), total=len(queries), desc="Retrieving and Re-ranking"):
    cos_scores = util.cos_sim(query_embedding, text_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, top_k).indices.tolist()

    cross_inputs = [(query, texts[i]) for i in top_k_indices]
    cross_scores = cross_encoder.predict(cross_inputs)

    best_idx = top_k_indices[int(torch.tensor(cross_scores).argmax())]
    pred.append(pages[best_idx])

query_df["answer"] = pred
query_df[["ID", "answer"]].to_csv("HW2_pred.csv", index=False)

print("Prediction completed and saved to HW2_pred.csv.")

