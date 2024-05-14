import boto3
import json
from dotenv import load_dotenv
import os
import requests


# loading in variables from .env file
load_dotenv()

# instantiating the Bedrock client, and passing in the CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))

bedrock = boto3.client('bedrock-runtime', 'us-east-1')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime','us-east-1')

# getting the knowledge base id from the env variable
knowledge_base_id = os.getenv('knowledge_base_id')  

def get_contexts(query, kbId, numberOfResults=5):
    """
    This function takes a query, knowledge base id, and number of results as input, and returns the contexts for the query.
    :param query: This is the natural language query that is passed in through the app.py file.
    :param kbId: This is the knowledge base id that is gathered from the .env file.
    :param numberOfResults: This is the number of results that are returned from the knowledge base.
    :return: The contexts for the query.
    """
    # getting the contexts for the query from the knowledge base
    results = bedrock_agent_runtime.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults
            }
        }
    )
    #  creating a list to store the contexts
    contexts = []
    #   adding the contexts to the list
    for retrievedResult in results['retrievalResults']: 
        contexts.append(retrievedResult['content']['text'])
    #  returning the contexts
    return contexts


def answer_query(user_input):
    """
    This function takes the user question, queries Amazon Bedrock KnowledgeBases for that question,
    and gets context for the question.
    Once it has the context, it calls the LLM for the response
    :param user_input: This is the natural language question that is passed in through the app.py file.
    :return: The answer to your question from the LLM based on the context from the Knowledge Bases.
    """
    # Setting primary variables, of the user input
    userQuery = user_input
    # getting the contexts for the user input from Bedrock knowledge bases
    userContexts = get_contexts(userQuery, knowledge_base_id)

    # Configuring the Prompt for the LLM
    # TODO: EDIT THIS PROMPT TO OPTIMIZE FOR YOUR USE CASE
    prompt_data = """
    You are a Question and answering assistant and your responsibility is to answer user questions based on provided context
    
    Here is the context to reference:
    <context>
    {context_str}
    </context>

    Referencing the context, answer the user question
    <question>
    {query_str}
    </question>
    """

    # formatting the prompt template to add context and user query
    formatted_prompt_data = prompt_data.format(context_str=userContexts, query_str=userQuery)

    # Configuring the model parameters, preparing for inference
    # TODO: TUNE THESE PARAMETERS TO OPTIMIZE FOR YOUR USE CASE
    # prompt = {
    #     "max_tokens": 4096,
    #     "temperature": 0.5,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": formatted_prompt_data
    #                 }
    #             ]
    #         }
    #     ]
    # }
    
    prompt =    {
        "prompt": formatted_prompt_data,
        "max_tokens" : 4096,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50
    }
    
    # formatting the prompt as a json string
    json_prompt = json.dumps(prompt)    

    # get useLocal from env 
    useLocal = os.getenv('use_Local')
    if useLocal == "True":
        response = call_llama_model(json_prompt)
        result = {
            "answer": response,
            "context": userContexts
        }
        return result
    else: 
        # invoking Claude3, passing in our prompt
        response = bedrock.invoke_model(body=json_prompt, modelId="mistral.mistral-large-2402-v1:0",
                                        accept="application/json", contentType="application/json")
        # getting the response from Claude3 and parsing it to return to the end user
        response_body = json.loads(response.get('body').read())

        # the final string returned to the end user
        answer = response_body['outputs'][0]['text']

        result = {
            "answer": answer,
            "context": userContexts
        }
        # returning the final string to the end user
        return result
    

def call_llama_model(prompt):
    url = "http://localhost:11434/api/generate"  # replace with your Ollama URL and port
    headers = {"Content-Type": "application/json"}
    data = {"model": "mistral", "prompt": prompt, "max_tokens": 60, "stream": False}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.content.decode("utf-8")  # decode the bytes object to a string
        response_data = json.loads(response_data)  # convert the string to a Python dictionary
        model_output = response_data.get("response")
        return model_output
    else:
        return None