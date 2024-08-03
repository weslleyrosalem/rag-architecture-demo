import subprocess
from flask import Flask, request

subprocess.check_call(['elyra-metadata', 'import', 'runtimes', '--directory', 'runtimes'])

app = Flask(__name__)


@app.route('/', methods = ['POST'])
def execute_pipeline():

    with open('./auto.pdf', 'wb') as file:
        file.write(request.data)

    subprocess.check_call(['elyra-pipeline', 'submit', 'pdf-to-xml-var.pipeline', '--runtime-config', 'odh_dsp'])

    result = 'pipeline_executed!'
    print(result)

    return result
