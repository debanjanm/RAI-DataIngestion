##------------------------------------------------------------------------------##
## Load and preprocess CSV data
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

# 1. Initialize and load the data with the 'description' field
loader = CSVLoader(
    file_path='arxiv_data/all_chunks.csv',
    content_columns=["chunk_text"],
    metadata_columns=["topic", "chunk_id", "pdf_name"]
)
data = loader.load()

# 2. Define a function to transform the documents
def remove_key_from_content(documents, key):
    new_documents = []
    for doc in documents:
        # Create a new Document object with the cleaned page_content
        new_documents.append(
            Document(
                page_content=doc.page_content.replace(f'{key}: ', ''),
                metadata=doc.metadata
            )
        )
    return new_documents

# 3. Apply the function to the loaded data
cleaned_data = remove_key_from_content(data, 'chunk_text')

# 4. Print the cleaned documents
# for doc in cleaned_data:
#     print(doc.metadata)


##------------------------------------------------------------------------------##
## Set up environment variables for LM Studio
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(
    model="text-embedding-qwen3-embedding-4b",
    check_embedding_ctx_length=False
)

# text = "LangChain is a framework for developing applications powered by language models."
# single_vector = embeddings.embed_query(text)
# print(single_vector)
# print(str(single_vector)[:100])  # Show the first 100 characters of the vector

# texts = ["Hello world", "LangChain with LM Studio", "Local embeddings are great!"]

# db = Chroma.from_texts(texts, embedding)
# results = db.similarity_search("hello world")

# for d in results:
#     print(d.page_content)

##------------------------------------------------------------------------------##
# ## Create and populate Chroma vector store with filtered documents    
# from langchain_chroma import Chroma
# from tqdm import tqdm

# vector_store = Chroma(
#     collection_name="arxiv",
#     embedding_function=embedding,
#     persist_directory="./chroma_db"
# )

# print(f"Number of cleaned documents: {len(cleaned_data)}")


# # Filter documents with topic = 'stat_ML'
# filtered_docs = [doc for doc in cleaned_data if doc.metadata.get("topic") == "stat.ML"]

# print(f"Number of filtered documents (topic='stat.ML'): {len(filtered_docs)}")

# # Loop with tqdm progress bar
# for doc in tqdm(filtered_docs, desc="Adding stat_ML documents"):
#     try:
#         vector_store.add_documents(
#             documents=[doc],
#             ids=[doc.metadata["chunk_id"]]
#         )
#     except Exception as e:
#         print(f"⚠️ Error adding document {doc.metadata.get('chunk_id', 'unknown')}: {e}")


# print("✅ All 'stat.ML' documents added to the vector store.")

##------------------------------------------------------------------------------##
# ## Load existing Chroma vector store and perform similarity search
loaded_vector_store = Chroma(
    collection_name="arxiv",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)
print(loaded_vector_store)


query = "Explain PopularityAdjusted Block Model (PABM)"
docs = loaded_vector_store.similarity_search(query, k=2)


print("--- Search Results from Loaded DB ---")
for doc in docs:
    print("-----------------------------------")
    print(f"Document: {doc.page_content}")
    print("-----------------------------------")

##------------------------------------------------------------------------------##
## Set up a LangChain pipeline with prompt template and LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# choose model name exactly as LM Studio exposes it (check LM Studio UI)
# llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", temperature=0.2)  

llm = ChatOpenAI(model="qwen/qwen3-4b-2507", temperature=0)  


template = "Answer the Question based on the context below.\n\nContext: {context}\n\nQ: {question}\nA:"
prompt = PromptTemplate.from_template(template)

# prompt = PromptTemplate(input_variables=["q"], template="Context: Q: {q}\nA:")
chain = prompt | llm | StrOutputParser()

# print(chain.run("Explain recursion in 2 sentences."))

print(chain.invoke({"context": docs, "question": "Explain PopularityAdjusted Block Model (PABM)"}))

##------------------------------------------------------------------------------##

# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI # Example LLM

# # Define the prompt template
# template = "Translate the following English text to {language}: '{text}'"
# prompt = PromptTemplate.from_template(template)

# # Format the prompt with specific values
# formatted_prompt = prompt.invoke({"language": "French", "text": "Hello, how are you?"})

# # Print the formatted prompt (for demonstration)
# print(formatted_prompt.to_string())

# # Example of using with an LLM (requires an LLM setup)
# # llm = OpenAI()
# # response = llm.invoke(formatted_prompt)
# # print(response)