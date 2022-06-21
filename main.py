import time
from flask import Flask, make_response, jsonify, request
# from itsdangerous import json
from kubernetes import client, config
from service import *
import threading
import json

TIME_TO_REPLICAS = 20
MAX_REPLICAS = 10
DEPLOYMENT_NAME = 'managed-app'

app = Flask(__name__)
servicer = Service("lstm_model/")


def scale(replica_count=1, deployment_name='test-deployment.yaml'):
    # replica_count = request.args.get("replica_count", default=-1, type=int)
    # deployment_name = request.args.get("deployment_name", default="", type=str)
    # print("scaling to", replica_count)
    # return replica_count
    with client.ApiClient() as api_client:
        api = client.AppsV1Api(api_client)
        manifest_part = [{'op': 'replace', 'path': '/spec/replicas', 'value': int(replica_count)}]
        k8s_api_response = api.patch_namespaced_deployment_scale(deployment_name, 'default', manifest_part)
        return int(k8s_api_response.spec.replicas)


# you should provie a name of the demplyment described in yaml file
def get_replica_count(deployment_name: str = 'test-deployment.yaml'):
    with client.ApiClient() as api_client:
        api = client.AppsV1Api(api_client)
        replica_response = api.list_namespaced_replica_set('default')
        count = -1
        for replica_set_info in replica_response.items:
            if deployment_name in replica_set_info.metadata.name:
                count = replica_set_info.spec.replicas
                break
        # Error if not found
        return count


@app.route("/send", methods=['POST'])
def add_movie():
    data = request.get_json()
    # print(data)
    try:
        if data['upload_time']: pass
    except KeyError:
        data = data['data']

    servicer.add_movie(data)
    return make_response(jsonify({'content': 'movie added'}), 200)


def start_flask():
    app.run(debug=True, host="0.0.0.0", port=5432)


def update_replicas():
    current_replicas = get_replica_count(DEPLOYMENT_NAME)
    while True:
        predicted_views = servicer.predict_max_stress() / 40
        required_replicas = predicted_views // TIME_TO_REPLICAS

        if required_replicas <= 0:
            required_replicas = 1
        elif required_replicas > MAX_REPLICAS:
            required_replicas = MAX_REPLICAS

        if current_replicas != required_replicas:
            current_replicas = scale(required_replicas, deployment_name=DEPLOYMENT_NAME)
        time.sleep(15)


if __name__ == "__main__":
    # config.load_kube_config()  # local version
    config.load_incluster_config()  # cluster version, needs extra privilege (k8s role manifest)

    scaling_thread = threading.Thread(target=update_replicas)
    scaling_thread.start()
    start_flask()
