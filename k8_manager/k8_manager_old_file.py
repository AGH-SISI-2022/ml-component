from flask import Flask, make_response, jsonify, request
from kubernetes import client, config

app = Flask(__name__)


@app.route("/scale", methods=['PUT'])
def scale():
    replica_count = request.args.get("replica_count", default=-1, type=int)
    deployment_name = request.args.get("deployment_name", default="", type=str)

    with client.ApiClient() as api_client:
        api = client.AppsV1Api(api_client)
        manifest_part = [{'op': 'replace', 'path': '/spec/replicas', 'value': replica_count}]
        k8s_api_response = api.patch_namespaced_deployment_scale(deployment_name, 'default', manifest_part)
        data = {"count": k8s_api_response.spec.replicas}
        return make_response(jsonify(data), 200)


@app.route("/replica-count/<deployment_name>", methods=['GET'])
def get_replica_count(deployment_name: str):
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


if __name__ == "__main__":
    # config.load_kube_config()   # local version
    config.load_incluster_config()    # cluster version, needs extra privilege (k8s role manifest)
    app.run(debug=True, host="0.0.0.0", port=5432)
