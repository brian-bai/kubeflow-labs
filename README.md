# kubeflow-labs
kubeflow labs on home server


## start kubeflow 
```
ssh user@hs

./minikube_start.sh

$ minikube config view
- cpus: 16
- memory: 32033

./minikube_expose.sh
```

## R notebook

- create notebook with image: kubeflownotebookswg/rstudio-tidyverse:v1.8.0
- create performance-report.Rmd
- knit pdf report
- need to install tinytex for pdf document generator
```{R}
tinytex::install_tinytex()
```

## Training Operator(Distributed training pilot)
- minikube host path mount
```
minikube mount /data/imagenet-mini:/data/imagenet-mini
```
- ansible-book for distributed training
```
ansible-playbook playbook/k8s-release-playbook.yaml
ansible-playbook playbook/k8s-train-playbook.yaml
ansible-playbook playbook/k8s-gather-log-playbook.yaml -e "gatherdir=cpu4-epoch10-run1" 
```