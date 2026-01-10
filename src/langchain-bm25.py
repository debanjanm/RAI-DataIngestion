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
for doc in cleaned_data:
    print(doc.metadata)

##------------------------------------------------------------------------------##
## Set up environment variables for LM Studio
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"

##------------------------------------------------------------------------------##

import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# filtered_docs = [doc for doc in cleaned_data if doc.metadata.get("topic") == "stat.ML"]
# documents = filtered_docs


documents = cleaned_data  # Use all documents

cleaned_data_topics = set(doc.metadata.get("topic") for doc in cleaned_data)
print(f"Topics in cleaned data: {cleaned_data_topics}")

# Initialize the BM25Retriever
bm25_retriever = BM25Retriever.from_documents(documents)

# Define the file path to save the retriever
file_path = "bm25_retriever_all.pkl"

# Save the BM25Retriever using pickle
with open(file_path, "wb") as f:
    pickle.dump(bm25_retriever, f)

print(f"BM25Retriever saved to {file_path}")

##------------------------------------------------------------------------------##
# import pickle
# from langchain_community.retrievers import BM25Retriever # Import is still needed for loading

# # Define the file path where the retriever is saved
# file_path = "bm25_retriever.pkl"

# # Load the BM25Retriever using pickle
# with open(file_path, "rb") as f:
#     loaded_bm25_retriever = pickle.load(f)

# print(f"BM25Retriever loaded from {file_path}")

# # You can now use the loaded_bm25_retriever
# # query = "lazy dog"
# query = "Explain PopularityAdjusted Block Model (PABM)"
# relevant_docs = loaded_bm25_retriever.invoke(query)
# for doc in relevant_docs:
#     print(doc.page_content)

##------------------------------------------------------------------------------##
# # Set up a LangChain pipeline with prompt template and LLM
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # choose model name exactly as LM Studio exposes it (check LM Studio UI)
# # llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", temperature=0.2)  

# llm = ChatOpenAI(model="qwen/qwen3-4b-2507", temperature=0)  


# template = "Answer the Question based on the context below.\n\nContext: {context}\n\nQ: {question}\nA:"
# prompt = PromptTemplate.from_template(template)

# # prompt = PromptTemplate(input_variables=["q"], template="Context: Q: {q}\nA:")
# chain = prompt | llm | StrOutputParser()

# # print(chain.run("Explain recursion in 2 sentences."))

# print(chain.invoke({"context": relevant_docs, "question": "Explain PopularityAdjusted Block Model (PABM)"}))

##------------------------------------------------------------------------------##
