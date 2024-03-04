from flask import Flask, request, jsonify
from flask_cors import CORS
# import ipynb_to_api.ipynb

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
os.environ["OPEN_API_KEY"] = "your_api_key" 

def rag_setup():
        # app = Flask(__name__)
        # These links contain information on refrigerators, dishwashers, transaction protocols, and return policies
        imp_links = [
            "https://www.partselect.com/",
            "https://www.partselect.com/shipping.aspx",
            "https://www.partselect.com/shipping.aspx" 
            "https://www.partselect.com/secure/Purchase.aspx",
            "https://www.partselect.com/Dishwasher-Parts.htm", 
            "https://www.partselect.com/Refrigerator-Parts.htm", 
            "https://www.partselect.com/Refrigerator-Models.htm", 
            "https://www.partselect.com/Dishwasher-Models.htm",
            "https://www.partselect.com/Repair/Dishwasher/", 
            "https://www.partselect.com/Repair/Refrigerator/", 
            "https://www.partselect.com/365-Day-Returns.htm"
        ]

        # Scraped data only on most popular refrigerator and dishwasher parts 
        file_paths = [
            "/Users/sarah_prakriti_peters/Documents/Instalily/data/fridge_data.csv",
            "/Users/sarah_prakriti_peters/Documents/Instalily/data/dishwasher_data.csv"
        ]

        csv_args = {"delimiter": ","}
        all_data = []

        print("Loading data")
        for file_path in file_paths:
            csv_loader = CSVLoader(file_path=file_path, csv_args=csv_args) 
            csv_data = csv_loader.load()
            all_data.extend(csv_data)

        web_loader = WebBaseLoader(imp_links)
        web_data = web_loader.load()
        all_data.extend(web_data) 

        # print("Splitting data")
        # Split the documents, and create chunks of information for embedding
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
        chunks = text_splitter.split_documents(all_data)

        client = weaviate.Client(embedded_options = EmbeddedOptions())

        # Create the vector database, for quicker retrieval
        vectorstore = Weaviate.from_documents(client = client, documents = chunks,
            embedding = OpenAIEmbeddings(openai_api_key="your_api_key"),by_text = False)

        retriever = vectorstore.as_retriever()

        # Set up prompt template to ensure answers are relevant
        template = """You are an assistant for question-answering tasks, and your goal is to be a customer support agent. 
        Use the following pieces of retrieved context to answer the question. 
        If there are any questions outside the scope of refrigerators, dishwashers, payments and returns, then please return an answer that says you cannot help with this query.
        If there is a question related to where to buy the entire refrigerator or dishwasher, then please mention that this website is primarily used for parts replacements
        If there is a question about return policies, please reroute them to the returns webpage - https://www.partselect.com/365-Day-Returns.htm

        Question: {question} 
        Context: {context} 
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Call GPT3.5 for each user query
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="your_api_key")

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
    app.run(debug=True, port=8080)
