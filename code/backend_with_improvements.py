# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders.csv_loader import CSVLoader 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
import warnings
warnings.filterwarnings("ignore")

# %%

def rag_setup():
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
        "https://www.partselect.com/365-Day-Returns.htm",
        "https://www.partselect.com/Dishwasher-Bearings.htm", 
        "https://www.partselect.com/Dishwasher-Brackets-and-Flanges.htm", 
        "https://www.partselect.com/Dishwasher-Caps-and-Lids.htm", 
        "https://www.partselect.com/Dishwasher-Circuit-Boards-and-Touch-Pads.htm", 
        "https://www.partselect.com/Dishwasher-Dishracks.htm", 
        "https://www.partselect.com/Dishwasher-Dispensers.htm", 
        "https://www.partselect.com/Dishwasher-Doors.htm", 
        "https://www.partselect.com/Dishwasher-Drawers-and-Glides.htm", 
        "https://www.partselect.com/Dishwasher-Ducts-and-Vents.htm", 
        "https://www.partselect.com/Dishwasher-Elements-and-Burners.htm", 
        "https://www.partselect.com/Dishwasher-Filters.htm", 
        "https://www.partselect.com/Dishwasher-Grilles-and-Kickplates.htm", 
        "https://www.partselect.com/Dishwasher-Handles.htm", 
        "https://www.partselect.com/Dishwasher-Hardware.htm", 
        "https://www.partselect.com/Dishwasher-Hinges.htm", 
        "https://www.partselect.com/Dishwasher-Hoses-and-Tubes.htm", 
        "https://www.partselect.com/Dishwasher-Insulation.htm", 
        "https://www.partselect.com/Dishwasher-Knobs.htm", 
        "https://www.partselect.com/Dishwasher-Latches.htm", 
        "https://www.partselect.com/Dishwasher-Legs-and-Feet.htm", 
        "https://www.partselect.com/Dishwasher-Manuals-and-Literature.htm", 
        "https://www.partselect.com/Dishwasher-Motors.htm", 
        "https://www.partselect.com/Dishwasher-Panels.htm", 
        "https://www.partselect.com/Dishwasher-Pumps.htm", 
        "https://www.partselect.com/Dishwasher-Racks.htm", 
        "https://www.partselect.com/Dishwasher-Seals-and-Gaskets.htm", 
        "https://www.partselect.com/Dishwasher-Sensors.htm", 
        "https://www.partselect.com/Dishwasher-Spray-Arms.htm", 
        "https://www.partselect.com/Dishwasher-Springs-and-Shock-Absorbers.htm", 
        "https://www.partselect.com/Dishwasher-Switches.htm", 
        "https://www.partselect.com/Dishwasher-Thermostats.htm", 
        "https://www.partselect.com/Dishwasher-Timers.htm", 
        "https://www.partselect.com/Dishwasher-Trays-and-Shelves.htm", 
        "https://www.partselect.com/Dishwasher-Trim.htm", 
        "https://www.partselect.com/Dishwasher-Valves.htm", 
        "https://www.partselect.com/Dishwasher-Wheels-and-Rollers.htm", 
        "https://www.partselect.com/Dishwasher-Wire-Plugs-and-Connectors.htm",
        "https://www.partselect.com/Refrigerator-Bearings.htm", 
        "https://www.partselect.com/Refrigerator-Blades.htm", 
        "https://www.partselect.com/Refrigerator-Brackets-and-Flanges.htm", 
        "https://www.partselect.com/Refrigerator-Caps-and-Lids.htm", 
        "https://www.partselect.com/Refrigerator-Circuit-Boards-and-Touch-Pads.htm", 
        "https://www.partselect.com/Refrigerator-Compressors.htm", 
        "https://www.partselect.com/Refrigerator-Deflectors-and-Chutes.htm", 
        "https://www.partselect.com/Refrigerator-Dispensers.htm", 
        "https://www.partselect.com/Refrigerator-Door-Shelves.htm", 
        "https://www.partselect.com/Refrigerator-Doors.htm", 
        "https://www.partselect.com/Refrigerator-Drawers-and-Glides.htm", 
        "https://www.partselect.com/Refrigerator-Drip-Bowls.htm", 
        "https://www.partselect.com/Refrigerator-Ducts-and-Vents.htm", 
        "https://www.partselect.com/Refrigerator-Electronics.htm", 
        "https://www.partselect.com/Refrigerator-Elements-and-Burners.htm", 
        "https://www.partselect.com/Refrigerator-Fans-and-Blowers.htm", 
        "https://www.partselect.com/Refrigerator-Filters.htm", 
        "https://www.partselect.com/Refrigerator-Grates.htm", 
        "https://www.partselect.com/Refrigerator-Grilles-and-Kickplates.htm", 
        "https://www.partselect.com/Refrigerator-Handles.htm", 
        "https://www.partselect.com/Refrigerator-Hardware.htm", 
        "https://www.partselect.com/Refrigerator-Hinges.htm", 
        "https://www.partselect.com/Refrigerator-Hoses-and-Tubes.htm", 
        "https://www.partselect.com/Refrigerator-Ice-Makers.htm", 
        "https://www.partselect.com/Refrigerator-Insulation.htm", 
        "https://www.partselect.com/Refrigerator-Knobs.htm", 
        "https://www.partselect.com/Refrigerator-Latches.htm", 
        "https://www.partselect.com/Refrigerator-Legs-and-Feet.htm", 
        "https://www.partselect.com/Refrigerator-Lights-and-Bulbs.htm", 
        "https://www.partselect.com/Refrigerator-Manuals-and-Literature.htm", 
        "https://www.partselect.com/Refrigerator-Motors.htm"
    ]

    # %%
    popular_fridge_links = []
    fridge_data = pd.read_csv( "/Users/sarah_prakriti_peters/Documents/Instalily/data/fridge_data.csv")
    for link in fridge_data['Product_link']:
        popular_fridge_links.append(''.join(["https://www.partselect.com", str(link.split()[0])]))

    popular_dishwasher_links = []
    dishwasher_data = pd.read_csv( "/Users/sarah_prakriti_peters/Documents/Instalily/data/dishwasher_data.csv")
    for link in dishwasher_data['Product_link']:
        popular_dishwasher_links.append(''.join(["https://www.partselect.com", str(link.split()[0])]))

    # %%
    imp_links.extend(popular_fridge_links)

    # %%
    imp_links.extend(popular_dishwasher_links)

    # %%
    len(imp_links)

    # %%
    all_data = []

    # Scraped data only on most popular refrigerator and dishwasher parts 
    # file_paths = [
    #     "/Users/sarah_prakriti_peters/Documents/Instalily/data/fridge_data.csv",
    #     "/Users/sarah_prakriti_peters/Documents/Instalily/data/dishwasher_data.csv"
    # ]

    # csv_args = {"delimiter": ","}

    # for file_path in file_paths:
    #     csv_loader = CSVLoader(file_path=file_path, csv_args=csv_args) 
    #     csv_data = csv_loader.load()
    #     all_data.extend(csv_data)

    web_loader = WebBaseLoader(imp_links)
    web_data = web_loader.load()
    all_data.extend(web_data) 
    client = weaviate.Client(embedded_options = EmbeddedOptions()) 

    # Split the documents, and create chunks of information for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
    chunks = text_splitter.split_documents(all_data)


    # %%
    # Create the vector database, for quicker retrieval
    vectorstore = Weaviate.from_documents(client = client, 
                                        documents = chunks,
                                        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                                        #   embedding = OpenAIEmbeddings(openai_api_key="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"), 
                                        by_text=False)

    retriever = vectorstore.as_retriever(
        # search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
        )

    # %%
    template = """
    As a customer support agent, your task is to address inquiries related to refrigerators, dishwashers, payments, and returns only. 
    If a question falls outside these categories, respond with a message indicating that the query cannot be addressed. 
    When asked about purchasing entire appliances, clarify that the website specializes in parts replacements. 
    For inquiries about return policies, direct users to the returns webpage at https://www.partselect.com/365-Day-Returns.htm. 
    Ensure responses are concise, limited to five sentences, and if you cannot find the answer, simply state that you don't know.

    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # %%
    llm = ChatOpenAI(model_name="gpt-4-turbo-preview", 
                    temperature=0, 
                    max_tokens = 1000,
                    openai_api_key="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")

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt
        | llm 
        | StrOutputParser() 
    )

    return rag_chain

rag_chain = rag_setup()

app = Flask(__name__)
CORS(app)
@app.route('/invoke_query', methods=['POST'])
def get_ai_message():
    try:
        user_query = request.json['text']['content']
        print(user_query)
        ai_message = rag_chain.invoke(user_query)
        return jsonify({"message": ai_message})
    except KeyError:
        # Handle missing 'userQuery' key in JSON request
        return jsonify({"error": "Missing 'userQuery' parameter"}), 400
    except Exception as e:
        # Handle other exceptions
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(port=8080)



