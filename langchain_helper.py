import os
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
import chromadb
# Set environment variables for API keys
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key
os.environ["COHERE_API_KEY"] = "H9RZGcNlhPwvaygVmRNxa86wycv0lp9MSv1t7zIv"  # Replace with your actual key

# Initialize Cohere Embeddla
cohere_embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0",  # Or your desired model
    user_agent="MyLangChainApp/1.0"
)

# Initialize LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

# ChromaDB client initialization
client = chromadb.Client()

# Collection name
collection_name = "faq_collection"

def create_vector_db():
    loader = CSVLoader(file_path='NEW_QA_WORKBOOK.csv', source_column="prompt")  # Correct path!
    data = loader.load()

    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents=data,
        embedding=cohere_embeddings,
        client=client,
        collection_name=collection_name
    )
    print(f"Chroma collection '{collection_name}' created.")

def get_qa_chain():
    # Load Chroma vector store
    vectordb = Chroma(client=client, collection_name=collection_name, embedding_function=cohere_embeddings)

    # Create retriever
    retriever = vectordb.as_retriever()

    # Prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__ == "__main__":
    # Check if collection exists (and create if needed)
    collection_exists = False
    try:
        client.get_collection(collection_name)
        collection_exists = True
    except chromadb.errors.InvalidCollectionException:
        pass

    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist. Creating...")
        create_vector_db()

    chain = get_qa_chain()
    response = chain.invoke({"query": "Do you have a JavaScript course?"})
    print(response["result"])