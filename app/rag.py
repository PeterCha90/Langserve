import os

from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
USE_BGE_EMBEDDING = True

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="PETER:latest")

# 필수 디렉토리 생성 @Mineru
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")


def embed_file():
    filename = "new_product_csv_info.csv"
    with open(f"./files/{filename}", "rb") as f:
        file_content = f.read()

    file_path = f"./.cache/files/{filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{filename}/")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(
        text_splitter=text_splitter)
    print("*********"*8)
    print("Embedding Starts...")

    # BGE Embedding: @Mineru
    model_name = "BAAI/bge-m3"
    # GPU Device 설정:
    # - NVidia GPU: "cuda"
    # - Mac M1, M2, M3: "mps"
    # - CPU: "cpu"
    model_kwargs = {
        "device": "cuda"
        # "device": "mps"
        # "device": "cpu"
    }
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    print("...")
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    vector_retriever = vectorstore.as_retriever()

    print("Embedding is Done!")
    print("*********"*8)
    return vector_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = embed_file()
RAG_PROMPT_TEMPLATE = """당신은 주어진 검색된 다음 문맥을 사용하여 질문에 답하는 친절한 AI입니다. 200자 이내로 간략하게 대답하고, 답을 모른다면 모른다고 답변하세요. 
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# 체인을 생성합니다.
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
