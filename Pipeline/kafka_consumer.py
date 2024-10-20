import os
import json
import random
import string
from kafka import KafkaConsumer
from kfp.client import Client

# Configuração do Kafka
KAFKA_BROKER = 'pdf-upload-kafka-bootstrap.openshift-operators.svc.cluster.local:9092'
KAFKA_TOPIC = 'pdf-upload'

# Configuração do KFP client
KFP_HOST = 'https://ds-pipeline-dspa-safra-ai.apps.rosa-5hxrw.72zm.p1.openshiftapps.com'
KFP_TOKEN = 'sha256~'
PIPELINE_FILE = 'rhoai-pdf-to-xml.yaml'

# Inicializa o consumidor Kafka
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def generate_random_run_name(base_name):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{base_name}-{random_suffix}"

def submit_pipeline(file_identifier):
    # Configura os argumentos do pipeline
    pipeline_arguments = {
        'file_identifier': file_identifier
    }
    # Cria o cliente KFP
    client = Client(
        host=KFP_HOST,
        existing_token=KFP_TOKEN
    )
    # Submete o pipeline com os argumentos
    run_name = generate_random_run_name("pdf-to-xml-safra-irpf")
    experiment_name = "pdf-to-xml-safra-irpf"
    result = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_FILE,
        arguments=pipeline_arguments,
        run_name=run_name,
        experiment_name=experiment_name
    )
    # Saída do resultado
    print(result)

# Consome mensagens do tópico Kafka
for message in consumer:
    record = message.value
    # Extrai o nome do arquivo sem extensão
    file_identifier = os.path.splitext(record.get('Key').split('/')[-1])[0]
    print(f'Received message: {record}')
    print(f'File Identifier: {file_identifier}')
    # Submete o pipeline com o identificador do arquivo
    submit_pipeline(file_identifier)