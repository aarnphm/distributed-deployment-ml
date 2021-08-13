<h2>distributed-deployment-ml</h2>
<i>Case study integrating GPU support for BentoML</i>
<hr>

## NOTES:

This no longer applies for BentoML releases after 1.0. Mainly for prototyping pre 1.0

## Docker images cycle releases

### <b>NVIDIA Triton Server</b>
- managed their docker images from a [entrypoint.sh](https://github.com/triton-inference-server/server/blob/main/nvidia_entrypoint.sh) and a [build.py](https://github.com/triton-inference-server/server/blob/main/build.py)
- drawbacks:
    - their buildscripts are mainly ```cp``` binary built to the model and also include a cuda-enabled images as their base layer
    - not elegant and hard to maintain for developers
<hr>

## Serving Framework v. BentoML

### <b>Serving with BentoML</b>

#### usage of `@env(docker_base_image="nvidia/cuda")`

- dependent on the image having `python` built in


after packing, edit Dockerfile as follows for GPU-supports:

```dockerfile
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04 as nvidia-cuda
...
COPY --from=nvidia-cuda /usr/local/cuda-11.0 /usr/local/cuda
COPY --from=nvidia-cuda /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/
# apparently tensorflow need this linked in order to use GPU
RUN ln /usr/local/cuda/lib64/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.11
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
...

```

### ONNX
_relevant files can be found under [`onnx`](./onnx)_

In order to run inference with GPU, users must use `onnxruntime-gpu` as the library will automatically allocate GPU
resources to run inference, fallback to CPU if needed.

User can check if they have GPU support with [`get_providers()`](https://github.com/microsoft/onnxruntime/blob/78a29aebbcbd0c3b6dab734f221e0f3bf1e24c97/onnxruntime/python/session.py#L49-L86):

```python
...
# for ONNXModelArtifacts, session=self.artifacts.model
cuda = "CUDA" in session.get_providers()[0] # True
```

### PyTorch

_relevant files can be found under [`pytorch`](./pytorch)_

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Tensorflow

_relevant files can be found under [`tf`](./tf)_

```python
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# or using `with` statement:
with tf.device("/GPU:0"):
    ...

```

- errors when running `yatai-start` 

<hr>

## Kubernetes

### Running GPU in a Kubernetes Cluster

- How do people usually use GPU in the wild?
- wtf is kubeflow?

setting up minikube https://minikube.sigs.k8s.io/docs/tutorials/nvidia_gpu/

[NVIDIA's device plugins](https://github.com/NVIDIA/k8s-device-plugin)

- driver has to be manually installed, referred to [here](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)

```shell
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

- check `kubeadm` init
- check `kubelet` systemd init

- `minikube start --driver=kvm2 --kvm-gpu --docker-opt {all the device opts}`

- in order to run with GPU we need to do PCI passthrough and this requires an unbound GPU. I might have to set this up later since my current GPU is bounded to Xorg session.

- configuring pods to consume GPUs
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-gpu-pod
spec:
  containers:
  - name: my-gpu-container
    image: nvidia/cuda:10.0-runtime-ubuntu18.04
    command: ["/bin/bash", "-c", "--"]
    args: ["while true; do sleep 600; done;"]
    resources:
      limits:
       nvidia.com/gpu: 2
```


### <b>Notes from NVIDIA docker container</b>

web ui -> choose a higher level api -> if not then we structure way to use lower api

overwrite -> proposal for docker images

recent updates from systemd re-architecture broke `nvidia-docker`, refers to [#1447](https://github.com/NVIDIA/nvidia-docker/issues/1447). This issue is confirmed to be in the [patched](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760189260) for **future releases**.

current workaround:

```shell
# for debian users one can disable cgroup hierarchy by adding to GRUB_CMDLINE_LINUX_DEFAULT="quiet systemd.unified_cgroup_hierarchy=0"

# for arch users change #no-cgroups=true under /etc/nvidia-container-runtime/config.toml

# one can just run the below command 
docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl ...

# or setup a docker-compose.yml and do
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-modeset:/dev/nvidia-modeset
  - /dev/nvidia-uvm:/dev/nvidia-uvm
  - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
```
