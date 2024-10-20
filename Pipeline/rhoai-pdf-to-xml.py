import os
import subprocess
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary dependencies
dependencies = [
    "sentence-transformers",
    "pymilvus",
    "openai==0.28",
    "langchain_community",
    "minio",
    "pymupdf",
    "Pillow",
    "pytesseract",
    "pandas",
    "langchain-milvus",
    "opencv-python",
    "pdf2image"  # Added pdf2image
]

for dep in dependencies:
    install(dep)

import fitz  # PyMuPDF
import re
import openai
from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
#from langchain_milvus import Milvus
from langchain_community.vectorstores import Milvus
import pytesseract
from pytesseract import Output
import pandas as pd
import logging
import urllib3
from pdf2image import convert_from_path  # For converting PDFs to images

# MinIO configuration
AWS_S3_ENDPOINT = "minio-api-minio.apps.rosa.poc-ai.9ewf.p3.openshiftapps.com"
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_S3_BUCKET = ""

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create an HTTP client with timeouts
http_client = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=5.0, read=10.0)
)

# Initialize the MinIO client with the custom HTTP client
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True,
    http_client=http_client
)

# Check if the script received the file identifier as an environment variable
file_identifier = os.getenv('file_identifier')
#file_identifier = "joao"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)
else:
    print(f"file_identifier: {file_identifier}")

# Download the PDF file from MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' is successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# Function to remove the watermark from the PDF
def remove_watermark_advanced(pdf_path, output_path):
    doc = fitz.open(pdf_path)

    for page in doc:
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            page.delete_image(xref)

        annots = page.annots()
        if annots:
            for annot in annots:
                annot_info = annot.info
                if "Watermark" in annot_info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        page.apply_redactions()

    doc.save(output_path)
    print(f"Watermark removed: {output_path}")

# Remove the watermark from the PDF
remove_watermark_advanced(file_path, output_pdf_path)

# New data extraction function
def extract_text_new(pdf_path):
    extracted_text = ''
    if os.path.exists(pdf_path):
        try:
            paginas_imagens = convert_from_path(pdf_path)
            for i, imagem_processada in enumerate(paginas_imagens):
                print(f"Processing page {i+1} of the PDF...")

                # Custom configuration for Tesseract OCR
                ocr_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1'
#                ocr_config = r'-c preserve_interword_spaces=1 --tessdata-dir /usr/share/tesseract/tessdata --oem 1 --psm 1'
#
                # Perform OCR
                ocr_data = pytesseract.image_to_data(imagem_processada, lang='por', config=ocr_config, output_type=Output.DICT)

                # Convert data into a DataFrame for analysis
                ocr_dataframe = pd.DataFrame(ocr_data)

                # Data cleaning: remove whitespaces and invalid texts
                cleaned_df = ocr_dataframe[(ocr_dataframe.conf != '-1') & (ocr_dataframe.text != ' ') & (ocr_dataframe.text != '')]

                # Sort text blocks vertically based on 'top' position
                sorted_block_numbers = cleaned_df.groupby('block_num').first().sort_values('top').index.tolist()

                for block_num in sorted_block_numbers:
                    current_block = cleaned_df[cleaned_df['block_num'] == block_num]  # Filter data for the current block
                    filtered_text = current_block[current_block.text.str.len() > 3]  # Select texts with more than 3 characters
                    avg_char_width = (filtered_text.width / filtered_text.text.str.len()).mean()  # Calculate the average character width
                    prev_paragraph, prev_line, prev_left_margin = 0, 0, 0  # Variables to control paragraphs, lines, and previous margins

                    for idx, line_data in current_block.iterrows():
                        # Change paragraph or line as needed
                        if prev_paragraph != line_data['par_num']:
                            extracted_text += '\n'
                            prev_paragraph = line_data['par_num']
                            prev_line = line_data['line_num']
                            prev_left_margin = 0
                        elif prev_line != line_data['line_num']:
                            extracted_text += '\n'
                            prev_line = line_data['line_num']
                            prev_left_margin = 0

                        # Calculate how many spaces to add for alignment
                        spaces_to_add = 0
                        if line_data['left'] / avg_char_width > prev_left_margin + 1:
                            spaces_to_add = int((line_data['left']) / avg_char_width) - prev_left_margin
                            extracted_text += ' ' * spaces_to_add  # Add necessary spaces to maintain alignment
                        extracted_text += line_data['text'] + ' '  # Add the text of the current line
                        prev_left_margin += len(line_data['text']) + spaces_to_add + 1  # Update the left margin for the next line

                    extracted_text += '\n'  # Add a new line at the end of each block

            return extracted_text
        except Exception as e:
            print(f"Error processing the PDF: {e}")
            return ''
    else:
        print(f"The file {pdf_path} was not found.")
        return ''

# Function to extract the contributor's name
def extract_contributor_name(text):
    patterns = [
        r'NOME:\s*([A-Z\s]+)\s*CPF:',
        r'Nome do Contribuinte:\s*([A-Z\s]+)\s*CPF:',
        r'Nome:\s*([A-Z\s]+)\s*Data de Nascimento:',
        r'Declarante:\s*([A-Z\s]+)\s*CPF:',
        r'Contribuinte:\s*([A-Z\s]+)\s*CPF:',
        r'IDENTIFICAÇÃO DO CONTRIBUINTE\s*Nome:\s*([A-Z\s]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None

# Function to convert Brazilian currency values to decimal
def convert_brazilian_currency_to_decimal_only(text):
    def replace_func(match):
        value = match.group(0)
        value = value.replace('.', '')
        value = value.replace(',', '.')
    text = re.sub(r'\b\d{1,3}(\.\d{3})*,\d{2}\b', replace_func, text)
    return text

# Function to extract group and code from the text
def extract_group_code_from_pdf(text):
    pattern = r'GRUPO\s*(\d{2})\s+.*\s+CÓDIGO\s*(\d{2})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None

# Function to preprocess the extracted text
def preprocess_text_with_group_code(text):
    text = convert_brazilian_currency_to_decimal_only(text)
    group, code = extract_group_code_from_pdf(text)
    return text, group, code

# Milvus configurations
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = "root"
MILVUS_PASSWORD = "Milvus"
MILVUS_COLLECTION = "irpf"

# OpenAI configurations
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

# Function to split the text
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

# Function to store text parts in Milvus
def store_text_parts_in_milvus(text_parts, pdf_file):
    for i, part in enumerate(text_parts):
        preprocessed_text, group, code = preprocess_text_with_group_code(part)
        metadata = {"source": pdf_file, "part": i, "group": group, "code": code}
        store.add_texts([preprocessed_text], metadatas=[metadata])

# Initialize Milvus
store = Milvus(
    embedding_function=embedding_function,
    connection_args={
        "host": "vectordb-milvus.milvus.svc.cluster.local", 
        "port": 19530,
        "user": "root",
        "password": "Milvus",
        "timeout": 30  # Ajuste o timeout conforme necessário
    },
    collection_name="irpf",
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)


# Process the PDF
pdf_path = output_pdf_path
text = extract_text_new(pdf_path)
contributor_name = extract_contributor_name(text)

# Check if the contributor's name was found
if not contributor_name:
    print("Error: Contributor's name not found. The script will exit.")
    sys.exit(1)

text_parts = split_text(text)
store_text_parts_in_milvus(text_parts, os.path.basename(pdf_path))

print("PDF processed and stored in Milvus successfully.")
print(f"Contributor name: {contributor_name}")

# Função para realizar a consulta no Milvus e gerar o XML
def query_information(query, contributor_name):
    documents = store.search(query=query, k=1, search_type="similarity")

    context_parts = []
    for doc in documents:
        context = doc.page_content
        group, code = extract_group_code_from_pdf(context)
        if group and code:
            context_parts.append(f"<GRUPO: {group} CÓDIGO: {code}>\n{context}")
        else:
            context_parts.append(context)

    context_combined = "\n".join(context_parts)

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        #temperature=0.3,  # Configuração para tornar a resposta mais determinística
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente financeiro avançado, experiente com profundo conhecimento em declaração de imposto de renda. Seu único objetivo é extrair informações do contexto fornecido e gerar respostas no formato XML. NUNCA interrompa uma resposta devido ao seu tamanho. Crie o XML com TODAS as informações pertinentes à sessão de bens e direitos, respeitando TODOS seus atributos e todos seus detalhes."
            },
            {
                "role": "user",
                "content": f"{context_combined}\n\nQuais são todos os bens e direitos, suas informações e seus detalhes, declarados por {contributor_name}? O resultado deve ser apresentado exclusivamente em XML com TODAS as características e detalhes de cada um dos bens, conforme exemplos abaixo:\n\n\
<?xml version=\"1.0\" ?>\n\
<SECTION Name=\"DECLARACAO DE BENS E DIREITOS\">\n\
    <TABLE>\n\
        <ROW No=\"1\">\n\
            <Field Name=\"GRUPO\" Value=\"01\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.445,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"342.342,00\"/>\n\
            <Field Name=\"InscricaoMunicipal(IPTU)\" Value=\"23423424\"/>\n\
            <Field Name=\"Logradouro\" Value=\"RUA QUALQUER\"/>\n\
            <Field Name=\"Numero\" Value=\"89\"/>\n\
            <Field Name=\"Complemento\" Value=\"COMPLEM 2\"/>\n\
            <Field Name=\"Bairro\" Value=\"BRASILIA\"/>\n\
            <Field Name=\"Municipio\" Value=\"BRASÍLIA\"/>\n\
            <Field Name=\"UF\" Value=\"DF\"/>\n\
            <Field Name=\"CEP\" Value=\"1321587\"/>\n\
            <Field Name=\"AreaTotal\" Value=\"345,0 m²\"/>\n\
            <Field Name=\"DatadeAquisicao\" Value=\"12/12/1993\"/>\n\
            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n\
            <Field Name=\"NomeCartorio\" Value=\"CARTORIO DE SÇNJJKLCDF ASLK SAKÇK SAÇKLJ SAÇLKS\"/>\n\
            <Field Name=\"Matricula\" Value=\"2344234\"/>\n\
        </ROW>\n\
        <ROW No=\"2\">\n\
            <Field Name=\"GRUPO\" Value=\"01\"/>\n\
            <Field Name=\"CODIGO\" Value=\"14\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"ETIAM EGET ORNARE DOLOR. UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"34.534,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"45.353,00\"/>\n\
            <Field Name=\"Logradouro\" Value=\"RUA DA CURVA\"/>\n\
            <Field Name=\"Numero\" Value=\"2342\"/>\n\
            <Field Name=\"Complemento\" Value=\"COMPLEMENTO 22320 KL-KI\"/>\n\
            <Field Name=\"Bairro\" Value=\"DISTRIVOT\"/>\n\
            <Field Name=\"Municipio\" Value=\"ATALAIA DO NORTE\"/>\n\
            <Field Name=\"UF\" Value=\"AM\"/>\n\
            <Field Name=\"CEP\" Value=\"23442-222\"/>\n\
            <Field Name=\"AreaTotal\" Value=\"23.423,0 m²\"/>\n\
            <Field Name=\"DatadeAquisicao\" Value=\"13/11/1987\"/>\n\
            <Field Name=\"RegistradonoCartorio\" Value=\"Sim\"/>\n\
            <Field Name=\"NomeCartorio\" Value=\"4DFDFCJKLNDFÇJO ÇKSJ S-LJK\"/>\n\
            <Field Name=\"Matricula\" Value=\"3423434\"/>\n\
        </ROW>\n\
        <ROW No=\"3\">\n\
            <Field Name=\"GRUPO\" Value=\"01\"/>\n\
            <Field Name=\"CODIGO\" Value=\"16\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"2.233.223,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"2.233.223,00\"/>\n\
            <Field Name=\"CEV/CNO\" Value=\"21.313.43133/12\"/>\n\
            <Field Name=\"Logradouro\" Value=\"LOGRADOURO 32323 SDDSÇKLNM SA-CL SDS\"/>\n\
            <Field Name=\"Numero\" Value=\"212\"/>\n\
            <Field Name=\"Complemento\" Value=\"COMPLEMENTO 3223DSDDS-SDL\"/>\n\
            <Field Name=\"Bairro\" Value=\"SDFSDS\"/>\n\
            <Field Name=\"Municipio\" Value=\"ALTO BOA VISTA\"/>\n\
            <Field Name=\"UF\" Value=\"MT\"/>\n\
            <Field Name=\"CEP\" Value=\"23432-424\"/>\n\
            <Field Name=\"AreaTotal\" Value=\"34.234,0 ha\"/>\n\
            <Field Name=\"DatadeAquisicao\" Value=\"\"/>\n\
        </ROW>\n\
        <ROW No=\"4\">\n\
            <Field Name=\"GRUPO\" Value=\"01\"/>\n\
            <Field Name=\"CODIGO\" Value=\"17\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"ETIAM EGET ORNARE DOLOR. UT QUIS ALIQUAM LEO.\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"21.343,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"21.343,00\"/>\n\
            <Field Name=\"Logradouro\" Value=\"LOGRADOUR W-CMO-SC-DO-23-90\"/>\n\
            <Field Name=\"Complemento\" Value=\"S-M-KMSDASDFKÇMSDKS\"/>\n\
            <Field Name=\"Bairro\" Value=\"CWFJNJ-SNMF\"/>\n\
            <Field Name=\"Municipio\" Value=\"ATALAIA DO NORTE\"/>\n\
            <Field Name=\"UF\" Value=\"AM\"/>\n\
            <Field Name=\"CEP\" Value=\"23424-242\"/>\n\
            <Field Name=\"AreaTotal\" Value=\"234.234,0 m²\"/>\n\
            <Field Name=\"DatadeAquisicao\" Value=\"\"/>\n\
        </ROW>\n\
        <ROW No=\"5\">\n\
            <Field Name=\"GRUPO\" Value=\"02\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"234.223,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"234.223,00\"/>\n\
            <Field Name=\"RENAVAM\" Value=\"21332132132\"/>\n\
        </ROW>\n\
        <ROW No=\"6\">\n\
            <Field Name=\"GRUPO\" Value=\"02\"/>\n\
            <Field Name=\"CODIGO\" Value=\"02\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"234.423,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"3.424.323,00\"/>\n\
            <Field Name=\"RegistrodeAeronave\" Value=\"242342343423432\"/>\n\
        </ROW>\n\
        <ROW No=\"7\">\n\
            <Field Name=\"GRUPO\" Value=\"02\"/>\n\
            <Field Name=\"CODIGO\" Value=\"03\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQU\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.424,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"23.424,00\"/>\n\
            <Field Name=\"RegistrodeEmbarcacao\" Value=\"WESSFG234DCD\"/>\n\
        </ROW>\n\
        <ROW No=\"8\">\n\
            <Field Name=\"GRUPO\" Value=\"02\"/>\n\
            <Field Name=\"CODIGO\" Value=\"04\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"45.635,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"345.234,00\"/>\n\
        </ROW>\n\
        <ROW No=\"9\">\n\
            <Field Name=\"GRUPO\" Value=\"02\"/>\n\
            <Field Name=\"CODIGO\" Value=\"99\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC A\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"345.234,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"345.234,00\"/>\n\
        </ROW>\n\
        <ROW No=\"10\">\n\
            <Field Name=\"GRUPO\" Value=\"03\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"ETIAM EGET ORNARE DOLOR. UT QUIS ALIQUAM LEO.\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.440,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"3.445.323,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"392.336.134-37\"/>\n\
            <Field Name=\"CNPJ\" Value=\"23.353.453/5353-43\"/>\n\
            <Field Name=\"NegociadosemBolsa\" Value=\"Sim\"/>\n\
            <Field Name=\"CodigoNegociacao\" Value=\"45353543\"/>\n\
        </ROW>\n\
        <ROW No=\"11\">\n\
            <Field Name=\"GRUPO\" Value=\"03\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\" IN SED LECTUS ODIO. ETIAM EGET ORNARE DOLOR. UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"23.454,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"234.243,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Dependente\"/>\n\
            <Field Name=\"CPF\" Value=\"233.352.471-59\"/>\n\
            <Field Name=\"CNPJ\" Value=\"34.555.435/3534-53\"/>\n\
            <Field Name=\"NegociadosemBolsa\" Value=\"Não\"/>\n\
        </ROW>\n\
        <ROW No=\"12\">\n\
            <Field Name=\"GRUPO\" Value=\"03\"/>\n\
            <Field Name=\"CODIGO\" Value=\"02\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"5.450,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"230,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"392.336.134-37\"/>\n\
            <Field Name=\"CNPJ\" Value=\"23424234\"/>\n\
        </ROW>\n\
        <ROW No=\"13\">\n\
            <Field Name=\"GRUPO\" Value=\"03\"/>\n\
            <Field Name=\"CODIGO\" Value=\"99\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALIQUA\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"234.234,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"2.342.340,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Dependente\"/>\n\
            <Field Name=\"CPF\" Value=\"822.764.521-61\"/>\n\
            <Field Name=\"CNPJ\" Value=\"23.423.423/4324-23\"/>\n\
        </ROW>\n\
        <ROW No=\"14\">\n\
            <Field Name=\"GRUPO\" Value=\"04\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UT QUIS ALIQUAM LEO. DONEC ALI\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"213.123,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"213,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"\"/>\n\
            <Field Name=\"CNPJ\" Value=\"4555453454355\"/>\n\
            <Field Name=\"Banco\" Value=\"117\"/>\n\
            <Field Name=\"Agencia\" Value=\"1233\"/>\n\
            <Field Name=\"Conta\" Value=\"123313-2\"/>\n\
        </ROW>\n\
        <ROW No=\"15\">\n\
            <Field Name=\"GRUPO\" Value=\"04\"/>\n\
            <Field Name=\"CODIGO\" Value=\"02\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"LOREM IPSUM VEHICULA MATTIS SOCIOSQU UT ERAT NEQUE FUSCE\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"456.534,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"8.989,0\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"416.228.090-84\"/>\n\
            <Field Name=\"CNPJ\" Value=\"45.746.413/0001-58\"/>\n\
        </ROW>\n\
        <ROW No=\"16\">\n\
            <Field Name=\"GRUPO\" Value=\"04\"/>\n\
            <Field Name=\"CODIGO\" Value=\"05\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"LOREM IPSUM PURUS PHARETRA RUTRUM\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"456.834,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"0,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"416.228.090-84\"/>\n\
        </ROW>\n\
        <ROW No=\"17\">\n\
            <Field Name=\"GRUPO\" Value=\"05\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"LOREM IPSUM DUIS NISI EGESTAS ID RUTRUM ALIQUAM\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"2.345.234,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"6.457.630,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"416.228.090-84\"/>\n\
        </ROW>\n\
        <ROW No=\"18\">\n\
            <Field Name=\"GRUPO\" Value=\"06\"/>\n\
            <Field Name=\"CODIGO\" Value=\"01\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"UIDEM QUIBUSDAM AT CONSEQUATUR OPTIO UT QUIA IPSUM.\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"34.543,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"0,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CNPJ\" Value=\"22.478.150/0001-48\"/>\n\
            <Field Name=\"Banco\" Value=\"237\"/>\n\
            <Field Name=\"Agência\" Value=\"3453\"/>\n\
            <Field Name=\"Conta\" Value=\"345-3\"/>\n\
            <Field Name=\"Titular CPF\" Value=\"906.395.600-24\"/>\n\
        </ROW>\n\
        <ROW No=\"19\">\n\
            <Field Name=\"GRUPO\" Value=\"06\"/>\n\
            <Field Name=\"CODIGO\" Value=\"10\"/>\n\
            <Field Name=\"DISCRIMINACAO\" Value=\"LOREM IPSUM ID NAM DUIS PORTTITOR CURABITUR VITAE MALESUADA.\"/>\n\
            <Field Name=\"SITUACAOANTERIOR\" Value=\"234,00\"/>\n\
            <Field Name=\"SITUACAOATUAL\" Value=\"0,00\"/>\n\
            <Field Name=\"Bemoudireitopertencenteao\" Value=\"Titular\"/>\n\
            <Field Name=\"CPF\" Value=\"416.228.090-84\"/>\n\
        </ROW>\n\
    </TABLE>\n\
</SECTION>"
            }
        ]
    )

    return response['choices'][0]['message']['content']

# Main function
def main():
    query = f"Quais são todos os bens e direitos suas informações, seus atributos e detalhes, declarados por {contributor_name}? Não interrompa a resposta devido ao seu tamanho, forneça a resposta com todo o contexto que tiver. Valores monetarios devem estar em portugues do brasil, moeda real brl."
    result = query_information(query, contributor_name)

    # Remove code formatting if present
    if result.startswith("```xml"):
        result = result[6:]  # Remove "```xml" from the start
    if result.endswith("```"):
        result = result[:-3]  # Remove "```" from the end

    # Remove unnecessary spaces at the ends
    result = result.strip()

    file_name = f'./{file_identifier}.xml'

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(result)

    print(f'File {file_name} has been successfully saved.')

    # Upload the XML file to MinIO
    object_name = f"{file_identifier}.xml"
    bucket_name = 'irpf-xml'
    try:
        client.fput_object(bucket_name, object_name, file_name)
        print(f"'{object_name}' is successfully uploaded as object to bucket '{bucket_name}'.")
    except S3Error as e:
        print("Error occurred: ", e)

# Execute the main script
if __name__ == "__main__":
    main()