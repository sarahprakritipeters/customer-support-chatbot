#!/usr/bin/env python
# coding: utf-8

# In[45]:


from flask import Flask, request, jsonify
from langchain_community.document_loaders.csv_loader import CSVLoader 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate 
from weaviate.embedded import EmbeddedOptions 
from langchain.prompts import ChatPromptTemplate 
from langchain.chat_models import ChatOpenAI 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import json
import warnings
import requests
warnings.filterwarnings("ignore") 
os.environ["OPEN_API_KEY"] = "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" 


# In[46]:


app = Flask(__name__)
# These links contain information on refrigerators, dishwashers, transaction protocols, and return policies
imp_links = [
    "https://www.partselect.com/", 
    "https://www.partselect.com/shipping.aspx",
    "https://www.partselect.com/shipping.aspx", 
    "https://www.partselect.com/secure/Purchase.aspx",
    "https://www.partselect.com/Dishwasher-Parts.htm", 
    "https://www.partselect.com/Refrigerator-Parts.htm", 
    "https://www.partselect.com/Refrigerator-Models.htm", 
    "https://www.partselect.com/Dishwasher-Models.htm",
    "https://www.partselect.com/Repair/Dishwasher/", 
    "https://www.partselect.com/Repair/Refrigerator/", 
    "https://www.partselect.com/365-Day-Returns.htm"
]

# print("Pulling for the 1000 links")
# for i in range(2,1424):
#     link1 = f"https://www.partselect.com/Refrigerator-Models.htm?start={i}"
#     link2 = f"https://www.partselect.com/Dishwasher-Models.htm?start={i}"
#     imp_links.append(link1)
#     imp_links.append(link2)

# Scraped data only on most popular refrigerator and dishwasher parts 
file_paths = [
    "/Users/sarah_prakriti_peters/Documents/Instalily/data/fridge_data.csv",
    "/Users/sarah_prakriti_peters/Documents/Instalily/data/dishwasher_data.csv"
]

csv_args = {"delimiter": ","}
all_data = []

# print("Loading data")
for file_path in file_paths:
    csv_loader = CSVLoader(file_path=file_path, csv_args=csv_args) 
    csv_data = csv_loader.load()
    all_data.extend(csv_data)

web_loader = WebBaseLoader(imp_links)
web_data = web_loader.load()
all_data.extend(web_data) 

# print("Splitting data")
# Split the documents, and create chunks of information for embedding
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100) 
chunks = text_splitter.split_documents(all_data)

client = weaviate.Client(embedded_options = EmbeddedOptions())

# Create the vector database, for quicker retrieval
vectorstore = Weaviate.from_documents(client = client, documents = chunks,
    embedding = OpenAIEmbeddings(openai_api_key="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"),by_text = False)

retriever = vectorstore.as_retriever()

# Set up prompt template to ensure answers are relevant
template = """You are an assistant for question-answering tasks, and your goal is to be a customer support agent. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use at most 5 sentences, and try to keep the answer concise.
If there are any questions outside the scope of refrigerators, dishwashers, payments and returns, then please return an answer that says you cannot help with this query.
If there is a question related to where to buy the entire refrigerator or dishwasher, then please mention that this website is primarily used for parts replacements
If there is a question about return policies, please reroute them to the returns webpage - https://www.partselect.com/365-Day-Returns.htm

Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Call GPT3.5 for each user query
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser() 
)


# In[47]:


@app.route('/invoke_query', methods=['POST'])
def invoke_query():
    data = request.get_json()
    query = data.get('query')
    result = rag_chain.invoke(query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8080)


# In[49]:


# # GET /convert
# req = json.loads(REQUEST)
# args = req['args']

# if 'angle' not in args:
#   print(json.dumps({'convertedAngle': None}))
# else:
#   # Note the [0] when retrieving the argument.
#   # This is because you could potentially pass multiple angles.
# #   angle = int(args['angle'][0])
# #   converted = math.radians(angle)
#   print(json.dumps({'convertedAngle': "abcd"}))


# In[ ]:




