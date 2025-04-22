import os
import pickle
import argparse
from math import ceil

from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def preprocess_pdf(
    pdf_path: str,
    cache_dir: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    caption_model: str = "microsoft/Phi-4-multimodal-instruct",
    batch_size: int = 8,
):
    os.makedirs(cache_dir, exist_ok=True)
    docs_path = os.path.join(cache_dir, "cached_docs.pkl")
    index_dir = os.path.join(cache_dir, "faiss_index")
    if os.path.exists(docs_path) and os.path.isdir(index_dir):
        print("Found existing cache; skipping preprocessing.")
        return

    # 1. Load PDF text + images
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    raw_images = convert_from_path(pdf_path, dpi=150)

    image_map = {str(i + 1) : img.convert("RGB") for i, img in enumerate(raw_images)}

    images = []
    for doc in docs:
        label = doc.metadata.get("page_label")
        images.append(image_map[label])

    # 2. Set up Phi‑4 multimodal
    processor = AutoProcessor.from_pretrained(
        caption_model,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        caption_model,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    ).cuda()
    # Load default generation config (avoid broken local file)
    generation_config = GenerationConfig.from_pretrained(
        caption_model,
        trust_remote_code=True
    )

    # A minimal user/assistant prompt wrapping each image
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    instruction = "Carefully caption the page image. Focus on the visual content only."
    single_prompt = (
        f"{user_prompt}<|image_1|>{instruction}{prompt_suffix}"
        f"{assistant_prompt}"
    )

    # 3. Batch‑caption all pages
    num_pages = len(docs)
    num_batches = ceil(num_pages / batch_size)
    captions = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_pages)
        batch_imgs = [images[i].convert("RGB") for i in range(start, end)]
        batch_prompts = [single_prompt] * len(batch_imgs)

        inputs = processor(
            text=batch_prompts,
            images=batch_imgs,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        out_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=256,
            use_cache=True,
            do_sample=True,
            top_p=0.9,
        )
        # strip off the prompt tokens
        out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        batch_caps = processor.batch_decode(out_ids, skip_special_tokens=True)
        captions.extend(batch_caps)

    # 4. Prepend captions to each Document
    for doc, cap in zip(docs, captions):
        doc.page_content = f"[Image caption: {cap}]\n\n" + doc.page_content

    # 5. Embed & FAISS‐index
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 6. Cache to disk
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    vectorstore.save_local(index_dir)

    print(f"Preprocessed {len(docs)} pages. Cache saved to {cache_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--cache_dir",
        default="rag_cache",
        help="Directory to store cached docs + FAISS index"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="How many pages to caption at once"
    )
    args = parser.parse_args()
    preprocess_pdf(
        args.pdf,
        args.cache_dir,
        batch_size=args.batch_size
    )

