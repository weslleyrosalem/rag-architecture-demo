#pip install minio pymupdf sentence-transformers pymilvus openai==0.28 langchain_community

import subprocess
import os
import sys

subprocess.run(["pip","install","sentence-transformers"])
subprocess.run(["pip","install","pymilvus"])
subprocess.run(["pip","install","openai==0.28"])
subprocess.run(["pip","install","langchain_community"])
subprocess.run(["pip","install","minio"])
subprocess.run(["pip","install","pymupdf"])


# Verifica se a variável de ambiente está definida
#file_identifier = "22_95658203015_2023_2024"
file_identifier = os.getenv('FILE_IDENTIFIER')
if not file_identifier:
    print("Error: The environment variable FILE_IDENTIFIER is not set.")
    sys.exit(1)

################################Minio-get.py################################
from minio import Minio
from minio.error import S3Error
import datetime

# MinIO client doesn't like URLs with procotol/schema, so use
# yourendpoint.com instead of https://yourtendpoint.com
AWS_S3_ENDPOINT = "minio-api-safra-ai.apps.rosa-6dgnl.ver6.p1.openshiftapps.com"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
AWS_S3_BUCKET = "irpf-2024"

# Create the MinIO client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True  # Set to True if you are using HTTPS
)

# List all objects
try:
    bucket = AWS_S3_BUCKET
    objects = client.list_objects(bucket)
    for object in objects:
        print(object.object_name, object._is_latest)
except S3Error as e:
    print("Error occurred: ", e)
    
# Download a file 
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' is successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)


################################Ingest-pdf-to-milvus################################

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF
import re
import openai

# Configuração do Milvus
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "safra_dir"

# Configuração da API OpenAI
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Inicialização do modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Função para gerar embeddings
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Função para pré-processar texto
def preprocess_text(text):
    # Remover pontuações e converter para minúsculas
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

# Função para dividir o texto em partes menores sem cortar palavras
def split_text(text, max_length=60000):  # Definindo um limite seguro abaixo do máximo permitido
    words = text.split()
    parts = []
    current_part = []

    current_length = 0
    for word in words:
        # Adiciona uma margem de segurança para garantir que não exceda o limite
        if current_length + len(word) + 1 <= max_length:
            current_part.append(word)
            current_length += len(word) + 1
        else:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_length = len(word) + 1
    
    # Adiciona a última parte
    if current_part:
        parts.append(" ".join(current_part))

    return parts

# Inicialização do Milvus com auto_id=True
store = Milvus(
    embedding_function=embedding_function,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

# Processar PDFs e armazenar no Milvus
pdf_folder_path = './'
for pdf_file in os.listdir(pdf_folder_path):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(text)
        text_parts = split_text(preprocessed_text)
        for i, part in enumerate(text_parts):
            store.add_texts([part], metadatas=[{"source": pdf_file, "part": i}])

print("PDFs processados e armazenados no Milvus com sucesso.")


################################Inference-file-model################################

# Inicialização do modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Função para gerar embeddings
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

store = Milvus(
    embedding_function=embedding_function,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

def query_information(query):
    documents = store.search(query=query, k=4, search_type="similarity")
    context = " ".join(doc.page_content for doc in documents)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an advanced, knowledgeable, and ethical financial assistant with deep knowledge in banking services, income tax return and taxes known as Safra Assistant. Your primary goal is to provide useful, accurate, and safe responses to any question asked. Use the context provided to give the most relevant and accurate answers possible."},
            {"role": "user", "content": f"{context}\n\n{query}"}
        ]
    )
    return response.choices[0].message["content"]

# Teste do sistema
query = """Quais são os bens e direitos declarados? O resultado deve ser apresentado em formato xml. Conforme exemplo:
<?xml version="1.0" ?>
<SECTION Name="DECLARACAO DE BENS E DIREITOS">
    <TABLE>
        <ROW No="1">
            <Field Name="GRUPO" Value="01"/>
            <Field Name="CODIGO" Value="01"/>
            <Field Name="DISCRIMINACAO" Value="NUNC A PELLENTESQUE EST. INTEGER CONGUE UT NISL NON SODALES. NUNC ET NEQUE NIBH. INTEGER EFFICITUR, IPSUM QUIS RUTRUM PELLENTESQUE, LIGULA MAURIS VIVERRA ORCI, VEL VENENATIS DOLOR ARCU EGET TORTOR. SED AC SEMPER NIBH. UT VITAE DIAM NEQUE. MAECENAS LAOREET DUI ID ERAT LACINIA FRINGILLA. FUSCE MAURIS VELIT, BIBENDUM AG ALIQUET ET, MATTIS EGET ERAT. NAM AT TRISTIQUE TELLUS. SUSPENDISSE SIT AMET HENDRERIT PURUS, SIT AMET EUISMOD DUI. IN SED LECTUS ODIO. ETIAM EGET ORNARE DOLOR. UT QUIS ALIQUAM LEO. DONEC ALIQUA 105 - BRASIL Inscrição Municipal (IPTU): 23423424 Logradouro: RUA QUALQUER Nº: 89 Comp.: COMPLEM 2 Bairro: BRASILIA Município: BRASÍLIA UF: DF CEP: 1321587 Área Total: 345,0 m² Data de Aquisição: 12/12/1993 Registrado no Cartório: Sim Nome Cartório: CARTORIO DE SÇQNJJKLÇDF Matrícula: 2344234 ASLK SAKÇK SAÇKLJ SAÇLKS"/>
            <Field Name="SITUACAOANTERIOR" Value="23.445,00"/>
            <Field Name="SITUACAOATUAL" Value="342.342,00"/>
            <Field Name="InscricaoMunicipal(IPTU)" Value="23423424"/>
            <Field Name="Logradouro" Value="RUA QUALQUER"/>
            <Field Name="No" Value="89"/>
            <Field Name="Comp." Value="COMPLEM 2"/>
            <Field Name="Bairro" Value="BRASILIA"/>
            <Field Name="Municipio" Value="BRASÍLIA"/>
            <Field Name="UF" Value="DF"/>
            <Field Name="CEP" Value="1321587"/>
            <Field Name="AreaTotal" Value="345,0 m²"/>
            <Field Name="DatadeAquisicao" Value="12/12/1993"/>
            <Field Name="RegistradonoCartorio" Value="Sim"/>
            <Field Name="NomeCartorio" Value="CARTORIO DE SÇQNJJKLÇDF ASLK SAKÇK SAÇKLJ SAÇLKS"/>
            <Field Name="Matricula" Value="2344234"/>
        </ROW>
"""
result = query_information(query)
#print(result)

# Especificando o nome do arquivo
file_name = f'./{file_identifier}.xml'

# Escrevendo o conteúdo em um arquivo
with open(file_name, 'w', encoding='utf-8') as file:
    file.write(result)

print(f'Arquivo {file_name} foi salvo com sucesso.')


################################Minio-put.py################################
from minio import Minio
from minio.error import S3Error

# Create the MinIO client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True  # Set to True if you are using HTTPS
)

# Upload a File 
file_path = f"./{file_identifier}.xml"
object_name = f"{file_identifier}.xml"
bucket_name = 'irpf-xml'
try:
    client.fput_object(bucket_name, object_name, file_path)
    print(f"'{object_name}' is successfully uploaded as object to bucket '{bucket_name}'.")
except S3Error as e:
    print("Error occurred: ", e)
