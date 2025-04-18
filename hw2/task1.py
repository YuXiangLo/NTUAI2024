from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LLM_MODEL = "microsoft/phi-2"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "Who is Yu Xiang Luo?"
CV_FILE = "resume.pdf"
DB_PATH = "db"

# ========== Step 1: build LLM ==========
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

def wo_RAG():
    print("\n🧪 [只用 LLM 回答]：")
    only_llm_response = llm.invoke(QUERY)
    print(only_llm_response)

def w_RAG():
    # ========== Step 2: build knowledge ==========
    loader = PyPDFLoader(CV_FILE)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS)
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH)
    retriever = vectordb.as_retriever()

    # ========== Step 3: build RAG chain ==========
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    result = chain.invoke({"input": QUERY})

    print("\n🧠 [使用 RAG 回答]：")
    print(result['answer'])

if __name__ == '__main__':
    wo_RAG()
    w_RAG()

