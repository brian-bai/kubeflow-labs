from istio_auth import get_istio_auth_session
from kfp import dsl
import kfp


@dsl.component
def add(a: float, b: float) -> float:
    '''Calculates sum of two arguments'''
    return a + b


@dsl.pipeline(
    name='Addition pipeline',
    description='An example pipeline that performs addition calculations.')
def add_pipeline(
    a: float = 1.0,
    b: float = 7.0,
):
    first_add_task = add(a=a, b=4.0)
    second_add_task = add(a=first_add_task.output, b=b)



KUBEFLOW_ENDPOINT = "http://192.168.124.28:8080"
KUBEFLOW_USERNAME = "user@example.com"
KUBEFLOW_PASSWORD = "12341234"
NAME_SPACE = "kubeflow-user-example-com"

auth_session = get_istio_auth_session(
    url=KUBEFLOW_ENDPOINT,
    username=KUBEFLOW_USERNAME,
    password=KUBEFLOW_PASSWORD
)

client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline", namespace=f"{NAME_SPACE}", cookies=auth_session["session_cookie"])

#print(client.list_experiments())

client.create_run_from_pipeline_func(
    add_pipeline, arguments={
        'a': 7.0,
        'b': 8.0
    })
