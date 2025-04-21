import argparse
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from transformers import AutoProcessor, BlipForConditionalGeneration
from pdf2image import convert_from_path


def create_rag_pipeline(
    pdf_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "microsoft/phi-2"
) -> RetrievalQA:
    # 1. Convert each PDF page both to text chunks *and* to images
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()           # text per page
    images = convert_from_path(pdf_path, dpi=150)  # PIL image per page

    # 2. Set up BLIP image‑captioner
    img_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    img_captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    # 3. For each page, generate a caption and prepend to the text
    for doc in docs:
        # metadata 'page' is 1‑indexed
        idx = doc.metadata["page"] - 1
        pil_img = images[idx]

        # Prepare and run caption
        inputs = img_processor(images=pil_img, return_tensors="pt").to("cuda")
        out = img_captioner.generate(**inputs)
        caption = img_processor.decode(out[0], skip_special_tokens=True)

        # Merge caption + original text
        doc.page_content = f"[Image caption: {caption}]\n\n" + doc.page_content

    # 4. Create embeddings (now based on caption+text)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"}
    )

    # 5. Build a FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 6. Only 1 page per query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 7. Load the LLM via HuggingFace pipeline
    text_gen = pipeline(
        "text-generation",
        model=llm_model,
        trust_remote_code=True,
        return_full_text=False,
        max_new_tokens=256,
        temperature=0.7,
        device=0,
    )
    llm = HuggingFacePipeline(pipeline=text_gen)

    # 6. Build the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


def query_pdf(qa_chain: RetrievalQA, question: str):
    """
    Query the RAG pipeline and return the page number.
    """
    result = qa_chain(question)
    source_docs = result.get("source_documents", [])

    # We're only retrieving k=1, so take the first page if present
    if source_docs:
        page_no = source_docs[0].metadata.get("page")
        print('----------------------[Question]----------------------')
        print(question)
        print('>>>>>>>>>>>>>>>>>>>>>>>[Spline]<<<<<<<<<<<<<<<<<<<<<<<<')
        print(page_no + 1)
        print('======================[Question]======================')
        return page_no + 1
    return None


def batch_query(qa_chain: RetrievalQA, in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    # Expect columns: "ID", "Question"
    results = []
    for _, row in df.iterrows():
        pid = row["ID"]
        q   = row["Question"]
        page = query_pdf(qa_chain, q)
        results.append({"ID": pid, "Answer": page})
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} answers to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG PDF Query Tool")
    parser.add_argument("pdf", help="Path to the PDF file to index")
    parser.add_argument(
        "--csv",
        help="Path to CSV input file (with columns ID,Question)",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output CSV file (with columns ID,Answer)",
        default="answers.csv"
    )
    args = parser.parse_args()

    qa = create_rag_pipeline(args.pdf)

    if args.csv:
        batch_query(qa, args.csv, args.output)
    else:
        print("PDF indexed. You can now ask questions. Type 'exit' to quit.")
        while True:
            q = input("\nEnter your question: ")
            if q.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            page = query_pdf(qa, q)
            if page is not None:
                print(f"Source page: {page}")
            else:
                print("Source page not found.")

