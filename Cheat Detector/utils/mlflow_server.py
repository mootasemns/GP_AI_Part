import os
import subprocess
from config import readConfig 

def start_mlflow_server(host, port, artifact_dir):
    """Start the MLflow server.

    Args:
        host (str): Host address for the MLflow server.
        port (int): Port number for the MLflow server.
        artifact_dir (str): Directory for storing MLflow artifacts.

    Returns:
        None
    """
    # Command to start the MLflow server
    command = [
        'mlflow', 'server',
        '--host', host,
        '--port', str(port),
        '--backend-store-uri', artifact_dir,
        '--default-artifact-root', artifact_dir
    ]
    
    # Start the MLflow server
    subprocess.run(command)

if __name__ == '__main__':
    # Load MLflow configurations
    config = readConfig('config/mlflow_config.yaml')
    mlflow_config = config['mlflow']
    
    mlflow_host = mlflow_config['host']
    mlflow_port = mlflow_config['port']
    mlflow_artifact_dir = mlflow_config['artifact_dir']
    
    # Ensure the artifact directory exists
    os.makedirs(mlflow_artifact_dir, exist_ok=True)
    
    # Start the MLflow server
    print(f'Starting MLflow server at http://{mlflow_host}:{mlflow_port}')
    start_mlflow_server(mlflow_host, mlflow_port, mlflow_artifact_dir)
