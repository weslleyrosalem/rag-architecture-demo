import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
dependencies = [
    "sentence-transformers",
    "pymilvus",
    "openai==0.28",
    "langchain_community",
    "minio",
    "pymupdf"
]

for dep in dependencies:
    install(dep)

from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
import fitz  # PyMuPDF
import re
import openai

# Ensure the script receives the file identifier as an environment variable
file_identifier = os.getenv('file_identifier')
#file_identifier = "18_01782176543_2023_2024"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)
else:
    print(f"file_identifier: {file_identifier}")

# MinIO configuration
AWS_S3_ENDPOINT = "minio-api-safra-ai.apps.rosa-5hxrw.72zm.p1.openshiftapps.com"
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

# Download the PDF file from MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' is successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# Milvus configuration
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "safra_dir"

# OpenAI configuration
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def convert_brazilian_currency_to_decimal_only(text):
    def replace_func(match):
        value = match.group(0)
        # Remove pontos que separam milhares
        value = value.replace('.', '')
        # Substitui a vírgula decimal por um ponto
        value = value.replace(',', '.')
        return value

    text = re.sub(r'\b\d{1,3}(\.\d{3})*,\d{2}\b', replace_func, text)
    return text

def preprocess_text(text):
    text = convert_brazilian_currency_to_decimal_only(text)
    return text

def extract_contributor_name(text):
    match = re.search(r'NOME:\s*([A-Z\s]+)\s*CPF:', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r'Nome do Contribuinte:\s*([A-Z\s]+)\s*CPF:', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

def split_text(text, max_length=60000):
    words = text.split()
    parts = []
    current_part = []

    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_part.append(word)
            current_length += len(word) + 1
        else:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_length = len(word) + 1

    if current_part:
        parts.append(" ".join(current_part))

    return parts

store = Milvus(
    embedding_function=embedding_function,
    connection_args={"host": MILVUS_HOST, "port": 19530, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

pdf_folder_path = './'
for pdf_file in os.listdir(pdf_folder_path):
    if (pdf_file.endswith('.pdf') and pdf_file == f"{file_identifier}.pdf"):
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        contributor_name = extract_contributor_name(text)
        preprocessed_text = preprocess_text(text)
        text_parts = split_text(preprocessed_text)
        for i, part in enumerate(text_parts):
            store.add_texts([part], metadatas=[{"source": pdf_file, "part": i}])

print("PDFs processed and stored in Milvus successfully.")

print(f"Contributor name: {contributor_name}")

def query_information(query, contributor_name):
    documents = store.search(query=query, k=4, search_type="similarity")
    context = " ".join(doc.page_content for doc in documents)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um assistente financeiro avançado, experiente e ético com profundo conhecimento em serviços bancários, declaração de imposto de renda e impostos, conhecido como Assistente Safra. Seu único objetivo é fornecer respostas no formato XML, seguindo a estrutura do modelo fornecido, com os dados solicitados, não interrompa uma resposta devido ao seu tamanho grande, ao invés disso, forneça todos os dados pertinentes."},
            {"role": "user", "content": f"{context}\n\nQuais são os bens e direitos declarados por {contributor_name}? O resultado deve ser apresentado exclusivamente em xml, conforme exemplo:\n<?xml version=\"1.0\" ?>\n<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n    <TABLE>\n        <ROW No=\"1\">\n            <Field Name=\"GRUPO\" Value=\"01\"/>\n            <Field Name=\"CODIGO\" Value=\"01\"/>\n            <Field Name=\"DISCRIMINACAO\" Value=\"DONEC ALIQUA 105 - BRASIL Inscrição Municipal (IPTU): 23423424 Logradouro: RUA QUALQUER Nº: 89 Comp.: COMPLEM 2 Bairro: BRASILIA Município: BRASÍLIA UF: DF CEP: 1321587 Área Total: 345,0 m² Data de Aquisição: 12/12/1993 Registrado no Cartório: Sim Nome Cartório: CARTORIO DE SÇQNJJKLÇDF Matrícula: 2344234 ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.445,00\"/>\n            <Field Name=\"SITUACAOATUAL\" Value=\"342.342,00\"/>\n            <Field Name=\"InscricaoMunicipal(IPTU)\" Value=\"23423424\"/>\n            <Field Name=\"Logradouro\" Value=\"RUA QUALQUER\"/>\n            <Field Name=\"No\" Value=\"89\"/>\n            <Field Name=\"Comp.\" Value=\"COMPLEM 2\"/>\n            <Field Name=\"Bairro\" Value=\"BRASILIA\"/>\n            <Field Name=\"Municipio\" Value=\"BRASÍLIA\"/>\n            <Field Name=\"UF\" Value=\"DF\"/>\n            <Field Name=\"CEP\" Value=\"1321587\"/>\n            <Field Name=\"AreaTotal\" Value=\"345,0 m²\"/>\n            <Field Name=\"DatadeAquisicao\" Value=\"12/12/1993\"/>\n            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n            <Field Name=\"NomeCartorio\" Value=\"CARTORIO DE SÇQNJJKLÇDF ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n            <Field Name=\"Matricula\" Value=\"2344234\"/>\n        </ROW>\n    </TABLE>\n</SECTION>"}
        ]
    )
    return response.choices[0].message["content"]

if contributor_name:
    text = extract_text_from_pdf(file_path)
    query = f"Quais são os bens e direitos declarados por {contributor_name}"
    result = query_information(query, contributor_name)

    file_name = f'./{file_identifier}.xml'

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f'File {file_name} has been successfully saved.')

    # Upload a File
    file_path = f"./{file_identifier}.xml"
    object_name = f"{file_identifier}.xml"
    bucket_name = 'irpf-xml'
    try:
        client.fput_object(bucket_name, object_name, file_path)
        print(f"'{object_name}' is successfully uploaded as object to bucket '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred: ", e)
else:
    print("Contributor name not found.")