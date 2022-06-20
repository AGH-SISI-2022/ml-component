import time
from flask import Flask, make_response, jsonify, request
from itsdangerous import json
from kubernetes import client, config
from service import *
import threading
import json

TIME_TO_REPLICAS = 10000


app = Flask(__name__)
servicer = Service("lstm_model/")



def scale(replica_count = 1, deployment_name = 'test-deployment.yaml'):
    # replica_count = request.args.get("replica_count", default=-1, type=int)
    # deployment_name = request.args.get("deployment_name", default="", type=str)

    with client.ApiClient() as api_client:
        api = client.AppsV1Api(api_client)
        manifest_part = [{'op': 'replace', 'path': '/spec/replicas', 'value': replica_count}]
        k8s_api_response = api.patch_namespaced_deployment_scale(deployment_name, 'default', manifest_part)
        print(k8s_api_response.spec.replicas)


#you should provie a name of the demplyment described in yaml file
def get_replica_count(deployment_name : str = 'test-deployment.yaml'):
    with client.ApiClient() as api_client:
        api = client.AppsV1Api(api_client)
        replica_response = api.list_namespaced_replica_set('default')
        count = -1
        for replica_set_info in replica_response.items:
            if deployment_name in replica_set_info.metadata.name:
                count = replica_set_info.spec.replicas  # HOPEFULLY CORRECT FIELD
                break
        # Error if not found
        data = {"count": count}
        return make_response(jsonify(data), 200)


@app.route("/add_movie", methods=['PUT'])
def add_movie():
    data = request.get_json(force=True)['data']
    try: 
        if data['upload_time']: pass
    except KeyError:
        data = data['data']

    servicer.add_movie(data)
    return make_response(jsonify({'content': 'movie added'}), 200)

def start_flask():
    app.run(debug=True, host="0.0.0.0", port=5432)


if __name__ == "__main__":
    # config.load_kube_config()   # local version
    config.load_incluster_config()    # cluster version, needs extra privilege (k8s role manifest)
    
    flask_app = threading.Thread(target=start_flask, daemon=True)
    flask_app.start()

    while True:
        predicted_views = servicer.predict_max_stress()
        required_replicas = predicted_views // TIME_TO_REPLICAS
        scale(required_replicas)
        time.sleep(120)
