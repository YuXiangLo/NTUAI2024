# preprocess.py

import os
import pickle
import argparse

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoProcessor, BlipForConditionalGeneration
from pdf2image import convert_from_path


def preprocess_pdf(
    pdf_path: str,
    cache_dir: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    os.makedirs(cache_dir, exist_ok=True)
    docs_path = os.path.join(cache_dir, "cached_docs.pkl")
    index_dir = os.path.join(cache_dir, "faiss_index")

    if os.path.exists(docs_path) and os.path.isdir(index_dir):
        print("Found existing cache; skipping preprocessing.")
        return

    # 1. Load & split pages to text + images
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    images = convert_from_path(pdf_path, dpi=150)

    # 2. BLIP captioner
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    captioner = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cuda")

    # 3. Caption each page and prepend
    for doc in docs:
        idx = doc.metadata["page"] - 1
        inputs = processor(images=images[idx], return_tensors="pt").to("cuda")
        out = captioner.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        doc.page_content = f"[Image caption: {caption}]\n\n" + doc.page_content

    # 4. Create embeddings & build FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 5. Cache to disk
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    vectorstore.save_local(index_dir)

    print(f"Preprocessed {len(docs)} pages. Cache saved to {cache_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDF into FAISS cache")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--cache_dir",
        help="Directory to store cached docs + FAISS index",
        default="rag_cache"
    )
    args = parser.parse_args()
    preprocess_pdf(args.pdf, args.cache_dir)

