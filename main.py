from flask import Flask, request, jsonify, render_template
import os
from langchain.llms import HuggingFaceEndpoint
#from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from dataBase import Vector_store


message = """
Responde la siguiente pregunta usando el contexto, recurda no inventar datos, solo limítate a extraer información y sintetizarla, responde en español siempre:

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

hf_token = os.environ["HF_TOKEN"]




def query(model: str, token: str, query_text: str, k: int) -> str:
    llm = HuggingFaceEndpoint(repo_id=model, huggingfacehub_api_token=token)
    results = Vector_store.similarity_search_with_score(query_text, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(message)
    prompt = prompt_template.format(context=context_text, question=query_text)
    respuesta = llm.invoke(prompt)

    context2 = f"""
    'Pregunta': {query_text}\n
    'Respuesta': {respuesta}

    Responde la a la siguiente pregunta tenindo en cuenta la pregunta y la respesta anterior: ¿'Hubo información necesaria para contestar la pregunta? Contesta solo Sí o No'

    """

    prompt_template2 = ChatPromptTemplate.from_template(context2)
    prompt2 = prompt_template2.format(query_text=query_text, respuesta=respuesta)

    valid = llm.invoke(prompt2)

    if "no" in valid.lower():
        return respuesta 
    else:
        ficha = results[0][0].metadata['source'][10:]
        month = results[0][0].metadata['source'][-8:-6]
        year = results[0][0].metadata['source'][-12:-8]
        link = 'https://iieg.gob.mx/ns/wp-content/uploads/'+year+'/'+month+'/'+ficha
        respuesta = f'{respuesta}\nPor favor corrobora la información en el siguiente sitio: {link} '
        return respuesta


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    textinp = data['text']
    respuesta = query(model='mistralai/Mixtral-8x7B-Instruct-v0.1', token=hf_token, query_text=textinp, k=2)
    return jsonify({'response': respuesta})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
