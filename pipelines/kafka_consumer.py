import os
import subprocess
import json
from kafka import KafkaConsumer

# Configuração do Kafka
KAFKA_BROKER = 'pdf-upload-kafka-bootstrap.openshift-operators.svc.cluster.local:9092'
KAFKA_TOPIC = 'pdf-upload'

# Configuração do Elyra pipeline
PIPELINE_PATH = 'pdf-to-xml-var.pipeline'
RUNTIME_CONFIG = 'odh_dsp'

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def submit_pipeline(file_identifier):
    os.environ['file_identifier'] = file_identifier
    result = subprocess.run(['elyra-pipeline', 'submit', PIPELINE_PATH, '--runtime-config', RUNTIME_CONFIG], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

for message in consumer:
    record = message.value
    file_identifier = record.get('Key').split('/')[-1].split('.')[0]
    print(f'Received message: {record}')
    print(f'File Identifier: {file_identifier}')
    submit_pipeline(file_identifier)
