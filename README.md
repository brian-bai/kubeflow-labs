# kubeflow-labs
kubeflow labs on home server


## start kubeflow 
```
ssh user@hs
./minikube_start.sh
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