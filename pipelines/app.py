import subprocess
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/execute_pipeline', methods=['POST'])
def execute_pipeline():
    data = request.get_json()
    file_identifier = data.get('file_identifier')

    if not file_identifier:
        return jsonify({'error': 'file_identifier is required'}), 400

    # Set the environment variable for the script
    os.environ['file_identifier'] = file_identifier

    # Call the pipeline script
    result = subprocess.check_call([sys.executable, 'pipeline_script.py'])

    return jsonify({'result': 'pipeline_executed'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)