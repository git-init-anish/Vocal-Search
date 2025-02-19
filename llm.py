from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from transcribe import preprocess_and_transcribe
from webparser import fetch_and_process
from rag import store_and_retrieve
import ollama

def summarize_text(query,text):
    full_prompt = f"""Imagine you are an intelligent AI Search Engine tasked with answering user queries based on provided documents. 

    Instructions:
    You will be given set of documents with relevant information. You have to understand the information and try to answer the User's Question. 
    You have to be plain and clear, If you are unsure about the information just say it and prompt the user to ask more questions !
    The information you provide must be correct, clearly answer the query and optionally provide any more facts !
    You can choose to either answer the questions in a avery explanatory manner if you have more facts or just answer the question if not.


    Max: Char limit for each response should be 600 characters

    User Query:
    {query}

    ### **Relevant Document:**
    {text}

    ---

    Expected Response Format:
    1. If the document provides a clear answer**:
    - Answer the query in a well-structured, coherent manner.
    - Dont include in the response that you are referring to a document, Just proceed and answer the question.

    2. If the document contains partial information**:
    - Clearly state whatever information is available.
    - Suggest the User that if they want to ask anything else more !

    3. **If the document does not contain any relevant information**:
    - Respond politely:  
        "I'm sorry, but I could not find enough information in the provided document to accurately answer your question."

    - Avoid making assumptions or fabricating information.

    ---
    Example Input:
    User Query:
    "What are the benefits of using transformers in NLP?"


    Relevant Document:
    "Transformers are deep learning models that use self-attention mechanisms to process entire sequences efficiently. Unlike traditional RNNs, transformers do not rely on sequential data processing, making them faster and more scalable for NLP tasks."

    ---

    Example Output:
    "Transformers improve NLP by leveraging self-attention to process entire sequences efficiently. Unlike RNNs, they do not require sequential processing, making them faster and more scalable for handling large text corpora."

    ---

    Now, generate a response based on the given document and user query."""
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": full_prompt}])
    return response["message"]["content"]


def process_voice_query(audio_file):
      
    text_query = preprocess_and_transcribe(audio_file)
    text_query="What was the Palisades Fire ?"
    print(f"Query: {text_query}")
    docs = fetch_and_process(text_query)
    summary = summarize_text(text_query,docs)
    return summary



audio_path="/Users/anishkumariyer/Documents/Samsung/Record-016.wav"
summary= process_voice_query(audio_path)
print(summary)
