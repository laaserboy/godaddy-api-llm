"""
To do:

"""

import os

import json


from flask import request
from flask import Flask, render_template

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import GPT4All
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
    )


documents_gl = {}
embedding_function_gl = {}


# Get configs from file
with open("../godaddy_api_llm/conf/app_conf.json", "r", encoding='utf-8') as f:
    config = json.load(f)

# Get configs from env
config['model_dir'] = os.getenv("GD_MODEL_DIR")
config['secret_key'] = os.getenv("FLASK_SECRET_KEY")
config['rag_dir'] = os.getenv("GD_RAG_DIR")

app = Flask(__name__)
app.secret_key = config['app_secret_key']


FUNCTIONS_INTERNAL = '{}'
with open('../godaddy_api_llm/conf/functions_internal.json', 'r', encoding='utf-8') as f:
    FUNCTIONS_INTERNAL = f.read()

SESSION_DESCRIPTION = '''You are an internet expert.\nRespond only in JSON. Wrap all text in JSON.\n
        JSON should be of this form.
        {
          "function": "function_name",
          "parameters": {
             "key": "value"
          }
        }
        "parameters" is a required field.

        If you are asked, 'What is my order ID?', respond with this.

        {
           "function": "call_get_order_id",
           "parameters": {}
        }

        If you are asked to add a domain name such as duggles.com to cart, respond like this.
        {
           "function": "call_add_to_cart",
           "parameters": {
             "domain": "duggles.com"
           }
        }
        If you are asked to puchase the domains in the cart, respond like this. 
        {
           "function": "call_purchase",
           "parameters": {}
        }
        If you are asked to suggest a domain name related to Cupertino and coffee, respond like this. 
        {
           "function": "call_suggest",
           "parameters": {
               "domain": "cupertinocoffee.com"
           }
        }
        If you are asked what the curl call is to check availability of a domain name, dugglemama.com, respond like this.
        {
           "function": "call_curl",
           "parameters": {
               "description": "curl -X 'GET' 'https://api.test-godaddy.com/v1/domains/available?domain=dugglemama.com' -H 'accept: application/json' -H \\\"Authorization: sso-key $GD_KEY:$GD_SECRET\\\""
           }
        }
        If you are asked what the curl call is to suggest domains related to a domain name, dugglemantle.com, respond like this.
        {
           "function": "call_curl",
           "parameters": {
               "description": "curl -X 'GET' 'https://api.test-godaddy.com/v1/domains/suggest?query=dugglemantle.com' -H 'accept: application/json' -H \\\"Authorization: sso-key $GD_KEY:$GD_SECRET\\\""
           }
        }

        This is a summary of the functions you can call.
    ''' + FUNCTIONS_INTERNAL


def safe_json_loads(json_str, default_val):
    '''Return default value on error'''
    response = {}
    try:
        response = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        response = default_val
    return response

@app.route('/', methods=['GET'])
def get_root():
    '''Present test page'''
    return render_template('home.html')

@app.route('/prompt', methods=['GET', 'POST'])
def get_prompt_response():
    '''send one answer based on the prompt'''
    prompt = SESSION_DESCRIPTION
    model_name = 'mistral'
    post_dict = {}
    if request.method == 'GET':
        short_prompt = request.args.get('prompt', '')
    elif request.method == 'POST':
        post_read = request.stream.read().decode('utf-8')
        post_dict = json.loads(post_read)
        short_prompt = post_dict.get('prompt', 'How is the weather today?')
    use_rag = True
    if use_rag is True:
        # RAG
        # NEED mistral or openai, still
        response = load_rag_and_respond(short_prompt, SESSION_DESCRIPTION, config)
    else:
        # NO RAG
        prompt += short_prompt
        model_name = post_dict.get('model', 'mistral')
        response = get_ai_response(model_name, prompt)
    response_out = app.response_class(
        response=response,
        status=200,
        mimetype='application/json'
    )
    return response_out

def get_ai_response(model_name, question):
    '''Feed prompt to langchain model and get response'''
    template = """Question: {question}
        Answer: """
    model_res = '{}'
    llm = {}
    prompt = PromptTemplate(template=template, input_variables=["question"])
    local_path = (
        config['model_dir'] + '/mistral-7b-openorca.Q4_0.gguf'
    )
    callbacks = [StreamingStdOutCallbackHandler()]
    if model_name == 'open_ai':
        llm = ChatOpenAI()
    else:
        model_name = 'mistral'
    if model_name != 'open_ai':
        llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        config['llm_chain'] = llm_chain
        model_res = llm_chain.run(question)
    return model_res

def metadata_func(record: dict, metadata: dict) -> dict:
    """Load metadata from JSON"""
    metadata["path"] = record.get("path")
    metadata["parameters"] = record.get("parameters")
    metadata["summary"] = record.get("summary")
    return metadata

#import time

def load_rag_and_respond(user_prompt, session_description, conf):
    """Load JSON and answer questions"""
    global documents_gl
    global embedding_function_gl
    documents = {}
    loader = JSONLoader(
        file_path=f'{conf["rag_dir"]}/chat_subscriptions.json',
        jq_schema='.messages[].content',
        metadata_func=metadata_func,
        text_content=False)
    if documents_gl != {}:
        documents = documents_gl
    else:
        documents = loader.load_and_split()
        documents_gl = documents
    llm = GPT4All(
        model=f'{conf["model_dir"]}/mistral-7b-openorca.Q4_0.gguf'
    )
    session_description = '''You are an API expert. You only answer in JSON. Here is an example. When a user asks for an API to suggest domain names about blue cats, reply like this.
{
"path": "/v1/domain/suggest",
"parameters": {"query": "blue cats"}
}
When a user asks, I want the next domain notification for customerid eef323-eeff583-ffee473 and x-request-id 373733ef, respond like this.

{
  "path": "/v2/customers/eef323-eeff583-ffee473/domains/notifications",
  "parameters": {
    "X-Request-Id": "373733ef",
    "customerId": "eef323-eeff583-ffee473"
  } 
}

When a user requests, 'I want the subscription product groups for customer eef323-eeff583-ffee473 and subscriptionId 23ef-23ef-894', respond like this.

{
  "path": "/v1/subscriptions/23ef-23ef-894",
  "parameters": {
     "X-Shopper-Id": "eef323-eeff583-ffee473",
     "subscriptionId": "23ef-23ef-894"
  }
}
Do not provide any other commentary. Provide only JSON. '''
    session_description = session_description.replace('{', '{{')
    session_description = session_description.replace('}', '}}')
    #prompt = ChatPromptTemplate.from_messages([
    #    ("system", session_description),
    #    ("user", "{input}")
    #])
    chunk_size = 1500
    chunk_overlap = 150
    add_start_index = True

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index)

    docs_split = text_splitter.split_documents(documents)
    embedding_function = {}
    if embedding_function_gl == {}:
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_function_gl = embedding_function
    else:
        embedding_function = embedding_function_gl

    vectorstore = Chroma.from_documents(documents=docs_split, embedding=embedding_function)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # next line 15 seconds
    result = qa_chain({"question": session_description + user_prompt})
    answer = result['answer']
    # BECAUSE langchain cannot figure out a simple template
    answer = answer.replace('{{', '{')
    answer = answer.replace('}}', '}')
    return answer
