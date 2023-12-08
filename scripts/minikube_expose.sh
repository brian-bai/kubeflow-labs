echo 'start dashboard'
minikube dashboard &
echo 'start minikube proxy'
kubectl proxy --address='0.0.0.0' --accept-hosts='^*$' &
echo 'expose prometheus'
kubectl --namespace monitoring port-forward --address='0.0.0.0' svc/prometheus-k8s 9090 &
echo 'expose grafana'
kubectl --namespace monitoring port-forward --address='0.0.0.0' svc/grafana 3000 &
echo 'expose alertmanager'
kubectl --namespace monitoring port-forward --address='0.0.0.0' svc/alertmanager-main 9093 &
echo 'expose kubeflow dashboard'
kubectl port-forward --address 0.0.0.0 svc/istio-ingressgateway -n istio-system 8080:80 &