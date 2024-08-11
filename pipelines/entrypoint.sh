#!/bin/bash

# Configura diretórios e variáveis de ambiente
export ELYRA_METADATA_HOME=/tmp/.elyra/metadata
export ELYRA_RUNTIME_DIR=/tmp/.elyra/runtime

# Executa o consumidor Kafka
python /app/kafka_consumer.py