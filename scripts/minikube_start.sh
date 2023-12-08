minikube start --driver docker --container-runtime docker --gpus all --extra-config=kubelet.authentication-token-webhook=true --ext
ra-config=kubelet.authorization-mode=Webhook --extra-config=scheduler.bind-address=0.0.0.0 --extra-config=controller-manager.bind-a
ddress=0.0.0.0