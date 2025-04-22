import os
import re
import pickle
import argparse
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker  # :contentReference[oaicite:0]{index=0}
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder  # :contentReference[oaicite:1]{index=1}
from langchain.retrievers import ContextualCompressionRetriever


def load_qa_chain(
    cache_dir: str,
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    # embedding_model: str = "hkunlp/instructor-large",
    embedding_model: str = "sentence-transformers/stsb-roberta-large",
    cross_encoder_model: str = "cross-encoder/stsb-roberta-large",
    llm_model: str = "microsoft/phi-2"
) -> RetrievalQA:
    docs_path = os.path.join(cache_dir, "cached_docs.pkl")
    index_dir = os.path.join(cache_dir, "faiss_index")

    if not (os.path.exists(docs_path) and os.path.isdir(index_dir)):
        raise ValueError("Cache not found. Run preprocess.py first.")

    # load cached docs & vectorstore
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    # embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},
    )
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    # 1) FAISS retrieves top‑10
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2) wrap your HF cross‑encoder
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=cross_encoder_model,
        model_kwargs={"device": "cuda", "max_length": 512},
    )

    # 3) build the reranker compressor (return top‑1)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=1)

    # 4) two‑stage retriever: FAISS → cross‑encoder rerank
    retriever = ContextualCompressionRetriever(
        base_retriever=faiss_retriever,
        base_compressor=reranker,
    )
    # LLM setup
    txt_pipe = pipeline(
        "text-generation",
        model=llm_model,
        trust_remote_code=True,
        return_full_text=False,
        max_new_tokens=256,
        temperature=0.7,
        device=0
    )
    llm = HuggingFacePipeline(pipeline=txt_pipe)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


def batch_infer(qa_chain: RetrievalQA, in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    queries = [{"query": q} for q in df["Question"]]
    results = qa_chain.apply(queries)

    out = []
    for idx, res in enumerate(results):
        docs = res["source_documents"]
        page = docs[0].metadata.get("page_label") if docs else None
        out.append({"ID": df.at[idx, "ID"], "Answer": page})

    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} answers to {out_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference over cached RAG index")
    parser.add_argument("--cache_dir", default="cache_dir",
                        help="Directory where preprocess.py stored its cache")
    parser.add_argument("--csv", required=True,
                        help="Input CSV file with columns ID,Question")
    parser.add_argument("--output", default="answers.csv",
                        help="Output CSV file (with columns ID,Answer)")
    args = parser.parse_args()

    qa = load_qa_chain(args.cache_dir)
    batch_infer(qa, args.csv, args.output)
