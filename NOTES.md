## <b>NVIDIA Triton Server</b>

## <b>Tensorflow Serving</b>

## <b>TorchServe</b>

## <b>Notes from NVIDIA docker container</b>

recent updates from systemd rearchitecture broke `nvidia-docker`, refers to [#1447](https://github.com/NVIDIA/nvidia-docker/issues/1447). This issue is confirmed to be in the [patched](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760189260) for **future releases**.

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


## <b>Serving with BentoML</b>

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
resources to run inference, fallback to CPU if needed. User can check if they have GPU support with:

```python
...
# assume the user is serving ONNX model with ONNXModelArtifacts, our ONNX session would be 
# self.artifacts.model
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