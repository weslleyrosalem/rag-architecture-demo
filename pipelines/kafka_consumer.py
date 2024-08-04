from kafka import KafkaConsumer
import json
import requests

# Configurar o consumer Kafka
consumer = KafkaConsumer(
    'pdf-upload',
    bootstrap_servers='pdf-upload-kafka-bootstrap.openshift-operators.svc.cluster.local:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def process_message(message):
    file_identifier = message['Key'].split('/')[-1].replace('.pdf', '')
    payload = {'file_identifier': file_identifier}
    response = requests.post('http://localhost:8080/execute_pipeline', json=payload)
    print(response.text)

# Processar mensagens da fila
for message in consumer:
    process_message(message.value)
