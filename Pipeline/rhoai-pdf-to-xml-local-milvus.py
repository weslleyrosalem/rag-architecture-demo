import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Instalação das dependências necessárias
dependencies = [
    "sentence-transformers",
    "pymilvus",
    "openai==0.28",
    "langchain_community",
    "minio",
    "pymupdf",
    "google-cloud-vision",
    "Pillow"
]

for dep in dependencies:
    install(dep)

from google.cloud import vision
from google.oauth2 import service_account
import fitz  # PyMuPDF
import re
import openai
from minio import Minio
from minio.error import S3Error
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Milvus
#from langchain_milvus import MilvusVectorStore
from PIL import Image
import io


# Configuração do Google Cloud Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./poc-marvel-4a0ef9962da2.json"

# Configurações do MinIO
AWS_S3_ENDPOINT = "minio-api-minio.apps.rosa.poc-ai.9ewf.p3.openshiftapps.com"
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_S3_BUCKET = ""

# Criação do cliente MinIO
client = Minio(
    AWS_S3_ENDPOINT,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    secure=True  # Configurado para HTTPS
)

# Ensure the script receives the file identifier as an environment variable
#file_identifier = os.getenv('file_identifier')
file_identifier = "bernando"
if not file_identifier:
    print("Error: file_identifier environment variable is not set.")
    sys.exit(1)
else:
    print(f"file_identifier: {file_identifier}")

# Baixar o arquivo PDF do MinIO
object_name = f"{file_identifier}.pdf"
file_path = f"./{file_identifier}.pdf"
output_pdf_path = f"./{file_identifier}_no_watermark.pdf"

try:
    client.fget_object(AWS_S3_BUCKET, object_name, file_path)
    print(f"'{object_name}' is successfully downloaded to '{file_path}'.")
except S3Error as e:
    print("Error occurred: ", e)
    sys.exit(1)

# Função para remover a marca d'água do PDF
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

# Remover a marca d'água do PDF
remove_watermark_advanced(file_path, output_pdf_path)

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Função para realizar OCR no PDF usando Google Cloud Vision
def ocr_extract_text_from_pdf_gcv(pdf_path):
    try:
        # Inicializar o cliente do Google Cloud Vision
        client = vision.ImageAnnotatorClient()

        # Abre o PDF com PyMuPDF
        pdf_document = fitz.open(pdf_path)
        text = ""

        for i in range(len(pdf_document)):
            # Converte a página do PDF em uma imagem
            page = pdf_document[i]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Converte a imagem para bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Converte para o formato de imagem utilizado pelo Google Vision
            image = vision.Image(content=img_byte_arr)

            # Chama a API para realizar OCR
            response = client.text_detection(image=image)
            annotations = response.text_annotations

            if annotations:
                page_text = annotations[0].description
                text += page_text + "\n"
                print(f"Texto extraído da página {i + 1}: {page_text[:100]}...")  # Imprime os primeiros 100 caracteres

            if response.error.message:
                raise Exception(f'{response.error.message}')

        return text
    except Exception as e:
        print(f"Erro ao tentar realizar OCR no arquivo {pdf_path}: {e}")
        return ""

# Função para tentar extrair texto e fallback para OCR se necessário
def extract_text_with_fallback(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("Texto não encontrado com fitz. Iniciando OCR...")
        text = ocr_extract_text_from_pdf_gcv(pdf_path)
    return text

# Função para extrair nome do contribuinte
def extract_contributor_name(text):
    patterns = [
        r'NOME:\s*([A-Z\s]+)\s*CPF:',
        r'Nome do Contribuinte:\s*([A-Z\s]+)\s*CPF:',
        r'Nome:\s*([A-Z\s]+)\s*Data de Nascimento:',
        r'Declarante:\s*([A-Z\s]+)\s*CPF:',
        r'Contribuinte:\s*([A-Z\s]+)\s*CPF:',
        r'Identificação do Contribuinte\s*([A-Z\s]+)\s*CPF:'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None

# Configurações do Milvus
MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
MILVUS_USERNAME = ""
MILVUS_PASSWORD = ""
MILVUS_COLLECTION = "safra_dir"

# Configurações do OpenAI
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Inicialização do modelo de embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        return self.model.encode(query)

    def embed_documents(self, documents):
        return self.model.encode(documents)

embedding_function = EmbeddingFunctionWrapper(embedding_model)

# Extraindo texto do PDF (com fallback para OCR)
text = extract_text_with_fallback(output_pdf_path)
contributor_name = extract_contributor_name(text)

# Verificação se o nome do contribuinte foi encontrado
if not contributor_name:
    print("Erro: Nome do contribuinte não encontrado. O script será encerrado.")
    sys.exit(1)

# Função para converter valores monetários brasileiros para decimal
def convert_brazilian_currency_to_decimal_only(text):
    def replace_func(match):
        value = match.group(0)
        value = value.replace('.', '')
        value = value.replace(',', '.')
        return value

    text = re.sub(r'\b\d{1,3}(\.\d{3})*,\d{2}\b', replace_func, text)
    return text

# Função para extrair código e grupo do texto diretamente do PDF
def extract_group_code_from_pdf(text):
    pattern = r'GRUPO\s*(\d{2})\s+.*\s+CÓDIGO\s*(\d{2})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None

# Função para preprocessar o texto extraído
def preprocess_text_with_group_code(text):
    text = convert_brazilian_currency_to_decimal_only(text)
    group, code = extract_group_code_from_pdf(text)
    return text, group, code

# Função para dividir o texto
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

# Função para armazenar partes do texto no Milvus
def store_text_parts_in_milvus(text_parts, pdf_file):
    for i, part in enumerate(text_parts):
        preprocessed_text, group, code = preprocess_text_with_group_code(part)
        metadata = {"source": pdf_file, "part": i, "group": group, "code": code}
        store.add_texts([preprocessed_text], metadatas=[metadata])

# Inicialização do Milvus
store = Milvus(
    embedding_function=embedding_function,
    connection_args={"host": MILVUS_HOST, "port": 19530, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False,
    auto_id=True
)

# Processamento do PDF
pdf_folder_path = './'
for pdf_file in os.listdir(pdf_folder_path):
    if pdf_file.endswith('.pdf') and pdf_file == f"{file_identifier}_no_watermark.pdf":
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = extract_text_with_fallback(pdf_path)
        contributor_name = extract_contributor_name(text)
        if not contributor_name:
            print("Erro: Nome do contribuinte não encontrado. O script será encerrado.")
            sys.exit(1)
        text_parts = split_text(text)
        store_text_parts_in_milvus(text_parts, pdf_file)

print("PDFs processed and stored in Milvus successfully.")

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
        model="gpt-4o-2024-08-06",
        temperature=0.3,  # Configuração para tornar a resposta mais determinística
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


# Função principal
def main():
    if contributor_name:
        text = extract_text_with_fallback(output_pdf_path)
        query = f"Quais são todos os bens e direitos suas informações, seus atributos e detalhes, declarados por {contributor_name}?"
        result = query_information(query, contributor_name)

        # Remover formatação de código se existir
        if result.startswith("```xml"):
            result = result[6:]  # Remove "```xml" do início
        if result.endswith("```"):
            result = result[:-3]  # Remove "```" do final

        # Remove espaços desnecessários nas extremidades
        result = result.strip()

        file_name = f'./{file_identifier}.xml'

        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(result)

        print(f'File {file_name} has been successfully saved.')

        # Upload do arquivo XML para o MinIO
        file_path = f"./{file_identifier}.xml"
        object_name = f"{file_identifier}.xml"
        bucket_name = 'irpf-xml'
        try:
            client.fput_object(bucket_name, object_name, file_path)
            print(f"'{object_name}' is successfully uploaded as object to bucket '{bucket_name}'.")
        except S3Error as e:
            print("Error occurred: ", e)
    else:
        print("Erro: Nome do contribuinte não encontrado. O script será encerrado.")
        sys.exit(1)

# Executar o script principal
main()