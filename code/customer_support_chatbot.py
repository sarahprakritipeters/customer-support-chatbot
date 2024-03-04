# %%
import os
os.environ["OPEN_API_KEY"] = "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" 

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
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
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests 
import csv
import re

# %%
imp_links = ["https://www.partselect.com/", 
             "https://www.partselect.com/shipping.aspx",
            "https://www.partselect.com/shipping.aspx", 
            "https://www.partselect.com/secure/Purchase.aspx",
            "https://www.partselect.com/Dishwasher-Parts.htm", 
            "https://www.partselect.com/Refrigerator-Parts.htm",
            "https://www.partselect.com/Refrigerator-Models.htm", 
            "https://www.partselect.com/Dishwasher-Models.htm",
            "https://www.partselect.com/Repair/Dishwasher/", 
            "https://www.partselect.com/Repair/Refrigerator/",
            "https://www.partselect.com/365-Day-Returns.htm"]


# Pull related parts for each appliance
popular_fridge_url = "https://www.partselect.com/Refrigerator-Parts.htm"
popular_models = requests.get(popular_fridge_url)
soup = BeautifulSoup(popular_models.content, "html.parser")
part_section = soup.find('h2', id='ShopByPartType')
if part_section:
    parts_list = part_section.find_next('ul', class_='nf__links')
    if parts_list:
        parts = [part.text.replace(' ', '-') for part in parts_list.find_all('a')]
        parts_with_links = ["https://www.partselect.com/{}.htm".format(part.replace(' ', '-')) for part in parts]
        imp_links.extend(parts_with_links)


popular_dishwasher_url = "https://www.partselect.com/Dishwasher-Parts.htm"
popular_models = requests.get(popular_dishwasher_url)
soup = BeautifulSoup(popular_models.content, "html.parser")
part_section = soup.find('h2', id='ShopByPartType')
if part_section:
    parts_list = part_section.find_next('ul', class_='nf__links')
    if parts_list:
        parts = [part.text.replace(' ', '-') for part in parts_list.find_all('a')]
        parts_with_links = ["https://www.partselect.com/{}.htm".format(part.replace(' ', '-')) for part in parts]
        imp_links.extend(parts_with_links)

# %%
# len(imp_links)

# %%
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

# Split the documents, and create chunks of information for embedding
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
chunks = text_splitter.split_documents(all_data)

# Create the vector database, for quicker retrieval
vectorstore = Weaviate.from_documents(client = client, documents = chunks,
    embedding = OpenAIEmbeddings(openai_api_key="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"),by_text = False)

retriever = vectorstore.as_retriever()

# %%
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

# %%
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser() 
)

# %%
query = input()
rag_chain.invoke(query)

# %%
# test_cases = [
#  "The ice maker on my Whirlpool fridge is not working. How can I fix it?", 
#  "Is this part compatible with my WDT780SAEM1 model?",
#  "How can I install part number PS11752778?",
#  "How easy is the installation of a refrigerator drum bearing slide?",
#  "How do customers rate MWFP? Is it necessary to have that part if i use a dshwase",
#  "Do I need a lower dishrack wheel? What does it even fix?",
#  "What is a dishwasher belt drive used for? Give me the short answer please",
#  "Hw expsnvie is the dishwash fridge ice and water flter",
#  "Are whirpiol or kenmore fridges better rated?", 
#  "How can I pay for any part?",
#  "Are there options for credit card payments?",
#  "What are the shipping options?",
#  "Is shipping expensive?",
#  "Paypal?", 
#  "How much does a stove cost? are there any popular stove brands?",
#  "Can I buy a cycling thermostat?",
#  "What are some good refrigerator options? Are they very expensive?",
#  "But these are not refrigerator options -they're parts. Where can I buy the entire fridge?",
#  "What are the most popular refrigerator models?",
#  "What are the most commonly replaced dishwasher parts?",
#  "How to repair my refrigerator? What are some common refrigerator problems?",
#  "What are the return policies for a refrigerator?"]

# %%
# for query in test_cases:
#     print(query) 
#     print(rag_chain.invoke(query)) 
#     print("-" * 75)
#     print()

# %%



