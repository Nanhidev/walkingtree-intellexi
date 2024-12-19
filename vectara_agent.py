from llama_index.core import (
    SimpleDirectoryReader,
)

from llama_index.indices.managed.vectara import VectaraIndex

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
import os
import re
from dotenv import load_dotenv
load_dotenv()


VECTARA_CUSTOMER_ID = os.environ["VECTARA_CUSTOMER_ID"]
VECTARA_CORPUS_ID = os.environ["VECTARA_CORPUS_ID"]
VECTARA_API_KEY = os.environ["VECTARA_API_KEY"]
# print(os.getenv('OPENAI_API_KEY'))
# OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
def using_vectara_agent(filepaths,query):  
    file_names = [os.path.basename(str(file_name)) for file_name in filepaths]
    print("file_names:", file_names)

    files = {file_name: SimpleDirectoryReader(input_files=[f"UploadedFiles/{file_name}"]).load_data() for file_name in file_names}

    node_parser = SentenceSplitter()
    all_nodes = []

    for file_name in file_names:
        nodes = node_parser.get_nodes_from_documents(files[file_name])
        all_nodes.extend(nodes)
    
    # if not os.path.exists(combined_index_dir):
    combined_index = VectaraIndex.from_documents(all_nodes)
    # combined_index.storage_context.persist(persist_dir=combined_index_dir)
    # else:
         # Load the combined index from storage if it exists
        # storage_context = StorageContext.from_defaults(persist_dir=combined_index_dir)
        # combined_index = load_index_from_storage(storage_context, index_type=VectaraIndex)
    # combined_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=combined_index_dir))

    query_engine = combined_index.as_query_engine(similarity_top_k=5, n_sentences_before=3, n_sentences_after=3,vectara_query_mode="mmr")
    response = str(query_engine.query(query))
    
    print(response)
    return response

# # Example usage
# filepaths = ["UploadedFiles/MOS-one-pager.pdf", "UploadedFiles/AXIS (1).pdf"]
# query = "Explain about small caps funnd?"
# using_vectara_agent(filepaths, query)

# file_names=[]
# for file_name in filepaths:
#     file_names.append(os.path.basename(str(file_name)))
# # file_names = filepaths
# print("file_names:",file_names)

# files = {}
# for file_name in file_names:
#     files[file_name] = SimpleDirectoryReader(
#         input_files=[f"UploadedFiles/{file_name}"]
#     ).load_data()

# node_parser = SentenceSplitter()

# # this is for the baseline
# all_nodes = []
# indexes = []

# for idx, file_name in enumerate(file_names):
#     nodes = node_parser.get_nodes_from_documents(files[file_name])
#     print("nodes:",nodes)
#     all_nodes.extend(nodes)
#     file_name=str(file_name).split(".")[0]
#     match = re.search(r'\((\d+)\)', file_name)  # Search for a number within parentheses
#     if match:
#         number = match.group(1)  # Extract the number
#         file_name = re.sub(r'\s*\(\d+\)', '_', file_name) + number
#     if not os.path.exists(f"./UploadedFiles/{file_name}"):
#         # build vector index
#         # vector_index = VectorStoreIndex(nodes)
#         vector_index=VectaraIndex.from_documents(nodes)
#         vector_index.storage_context.persist(
#             persist_dir=f"./UploadedFiles/{file_name}"
#         )
#     else:
#         vector_index = load_index_from_storage(
#             StorageContext.from_defaults(persist_dir=f"./UploadedFiles/{file_name}"),
#         )

#     indexes.append(vector_index)

# response = str(indexes.as_query_engine(similarity_top_k=5, n_sentences_before=3, n_sentences_after=3).query(query))
# print(response)
# return response