# docker run on system76

```
ansible-playbook playbook/home-release-playbook.yaml

cd ~/workspace/gpuws
docker run -it -v $(pwd):/work -v ~/workspace/devdata/imagenet-mini:/imagenet-mini --gpus all --shm-size=2gb imgnet-local-test:1.0
# python imagenet-local.py -b 32 -a resnet18 --epochs 1 -p 1 >> local-run-e1p1b32resnet18.log
# python imagenet-local.py -a resnet18 --dist-url 'tcp://127.0.0.1:5432' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --epochs 1 -p 1 -b 32 >> local-1-gpu.log
```