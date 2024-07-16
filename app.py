from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import spacy

app = Flask(__name__)

# Load the Spanish model
nlp = spacy.load("es_core_news_md")

prueba = pd.read_excel("C:/Users/UsuarioPC/Documents/Chatbot/Contenido.xlsx")
prueba['nombre'] = prueba['nombre'].apply(lambda x: x.capitalize())

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def extract_text_after_keywords(text: str) -> list[str]:
    doc = nlp(text)
    matches = [str(token) for token in doc if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN']
    matches2 = ' '.join(matches)
    matches2 = [matches2]
    nosolas = ['precios', 'dato', 'precio', 'indicador', 'ficha', 'datos', 'indicadores', 'fichas', 'información', 'consulta', 'estudio', 'tablero', 'plataforma', 'estudios', 'consultas', 'industria', 'industrias']    
    matches = [m for m in matches if m not in nosolas]
    matchesf = matches + matches2
    return matchesf

def check_person(text: str) -> bool:
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents if ent.label_ == 'PER']
    if len(entities) == 0:
        return False
    else:
        return True

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sentsim(sentenceslist: list[str]) -> float:
    sentences = sentenceslist
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    cosi = torch.nn.CosineSimilarity(dim=0) 
    sim = cosi(sentence_embeddings[0], sentence_embeddings[1])
    return sim

def consulta(texto: str) -> list[str]:
    lista = []
    peticion = extract_text_after_keywords(texto)
    if len(peticion) == 0:
        return lista
    else:
        for e in range(len(peticion)):
            for k in range(len(prueba)):
                if peticion[e].lower() in prueba.iloc[k,3].lower():
                    lista.append(prueba.iloc[k,1])
                s = sentsim([prueba.iloc[k,3],peticion[e]])
                if s >= 0.6:
                    lista.append(prueba.iloc[k,1])
        lista = list(dict.fromkeys(lista))
        return lista

def recomendacion(texto: str) -> list[str]:
    res = []
    if check_person(texto) == True:
        res.append('Lo lamento, no contamos con la información que se solicita')
    else:
        lista = consulta(texto)
        if len(lista) == 0:
            res.append('Lo lamento, no entiendo tu consulta, pero con gusto puedes preguntarme de nuevo, te prometo haré un mayor esfuerzo')
        else:
            for n in lista:
                tipo = prueba.set_index('nombre').loc[n]['tipo']
                link = prueba.set_index('nombre').loc[n]['link']
                if isinstance(tipo, str) == False:
                    conjunto = list(zip(tipo, link))
                    for c in range(len(conjunto)):
                        t = conjunto[c][0]
                        l = conjunto[c][1]
                        if t[-1] == 'a':
                            if n[-1] == 'a' or n[-2:] == 'as' or n[-2:] == 'ón':
                                res.append('Te recomiendo consultar la {} sobre la {} en el siguiente link: {}'.format(t, n, l))
                            else:
                                res.append('Te recomiendo consultar la {} sobre el {} en el siguiente link: {}'.format(t, n, l))
                        else:
                            if n[-1] == 'a' or n[-2:] == 'as' or n[-2:] == 'ón':
                                res.append('Te recomiendo consultar el {} sobre la {} en el siguiente link: {}'.format(t, n, l))
                            else:
                                res.append('Te recomiendo consultar el {} sobre el {} en el siguiente link: {}'.format(t, n, l))
                else:
                    if tipo[-1] == 'a':
                        if n[-1] == 'a' or n[-2:] == 'as' or n[-2:] == 'ón':
                            res.append('Te recomiendo consultar la {} sobre la {} en el siguiente link: {}'.format(tipo, n, link))
                        else:
                            res.append('Te recomiendo consultar la {} sobre el {} en el siguiente link: {}'.format(tipo, n, link))
                    else:
                        if n[-1] == 'a' or n[-2:] == 'as' or n[-2:] == 'ón':
                            res.append('Te recomiendo consultar el {} sobre la {} en el siguiente link: {}'.format(tipo, n, link))
                        else:
                            res.append('Te recomiendo consultar el {} sobre el {} en el siguiente link: {}'.format(tipo, n, link))
    return res

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    textinp = data['text']
    respuesta = recomendacion(textinp)
    return jsonify({'response': respuesta})

if __name__ == '__main__':
    app.run(debug=True)

