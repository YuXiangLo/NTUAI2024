import re
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ====== Configuration ======
LLM_MODEL = "microsoft/phi-2"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "Who is Yu Xiang Luo?"
CV_FILE = "resume.pdf"

# ====== Build LLM & Embedding ======
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="cuda")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS)

# ====== Prompt & QA Chain ======
system_prompt = (
    "Use the given context to answer the question. "
    "Today is 2025/4/20. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, prompt)

# ====== Helpers ======
def dedupe(docs, threshold=0.9, topk=3):
    """
    Remove near-duplicate documents based on cosine similarity of embeddings.
    """
    filtered = []
    seen_embs = []
    # compute embeddings for each candidate doc
    contents = [doc.page_content for doc in docs]
    embs = embedding.embed_documents(contents)
    for doc, emb in zip(docs, embs):
        # check similarity against already kept embeddings
        is_dup = any(
            np.dot(emb, s) / (np.linalg.norm(emb) * np.linalg.norm(s)) > threshold
            for s in seen_embs
        )
        if not is_dup:
            filtered.append(doc)
            seen_embs.append(emb)
        if len(filtered) >= topk:
            break
    return filtered

# section-aware splitters
splitters = {
    "default": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    "section": RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "Education", "Experience", "Projects", "Technical Skills"]
    )
}

# ====== Experiment Runner ======
def run_experiment(name, split_type, search_type, k, fetch_k=None):
    print(f"\n=== Experiment: {name} ===")
    # Load pages
    loader = PyPDFLoader(CV_FILE)
    pages = loader.load_and_split()
    # Strip header (remove resume header boilerplate)
    for page in pages:
        page.page_content = re.sub('Yu Xiang Luo', '', page.page_content)
    # Split into chunks
    splitter = splitters[split_type]
    docs = splitter.split_documents(pages)
    # Build in-memory vector store
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding)
    # Configure retriever
    search_kwargs = {"k": k}
    if fetch_k is not None:
        search_kwargs["fetch_k"] = fetch_k
    retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    # Retrieve
    candidate_docs = retriever.get_relevant_documents(QUERY)
    # Optionally dedupe MMR outputs
    if search_type == "mmr" and fetch_k and fetch_k > k:
        candidate_docs = dedupe(candidate_docs, threshold=0.9, topk=k)
    # Generate answer
    result = qa_chain.invoke({"input": QUERY, "context": candidate_docs})
    print(result)

def run_no_chunk():
    print("\n=== Experiment: No-chunk Stuff (full resume) ===")
    loader = PyPDFLoader(CV_FILE)
    pages = loader.load_and_split()
    for page in pages:
        page.page_content = re.sub('Yu Xiang Luo', '', page.page_content)
    full_text = "\n\n".join(page.page_content for page in pages)
    docs = [Document(page_content=full_text)]
    result = qa_chain.invoke({"input": QUERY, "context": docs})
    print(result)

if __name__ == '__main__':
    run_no_chunk()

    experiments = [
        {"name": "MMR (k=3, fetch_k=3, default-split)", "split_type": "default", "search_type": "mmr", "k": 3, "fetch_k": 3},
        {"name": "MMR (k=3, fetch_k=10, default-split)", "split_type": "default", "search_type": "mmr", "k": 3, "fetch_k": 10},
        {"name": "MMR (k=3, fetch_k=10, section-split)", "split_type": "section", "search_type": "mmr", "k": 3, "fetch_k": 10},
        {"name": "Similarity (k=3, default-split)", "split_type": "default", "search_type": "similarity", "k": 3},
        {"name": "Similarity (k=3, section-split)", "split_type": "section", "search_type": "similarity", "k": 3},
    ]
    for exp in experiments:
        run_experiment(**exp)

