#!/usr/bin/env python3
import os
import sys
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
import constants

os.environ["OPENAI_API_KEY"] = constants.API_KEY

CACHE = False
# query = sys.argv[1]
loader = PyPDFLoader("doc.pdf")
# loader = DirectoryLoader(".", glob="*.txt")

if CACHE and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorstoreIndexCreator(vectorstore=vectorstore).from_loaders([loader])
else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model="gpt-3.5-turbo"),
#     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
# )
chain = ConversationalRetrievalChain.from_llm(
  llm =ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

query = None
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])
  chat_history.append((query, result['answer']))
  query = None


print(chain.run(query))
