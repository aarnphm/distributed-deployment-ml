<b>NVIDIA Triton Server</b>

<b>Tensorflow Serving</b>

<b>TorchServe</b>

<b>Notes from NVIDIA docker container</b>

recent updates from systemd rearchitecture broke `nvidia-docker`, refers to [#1447](https://github.com/NVIDIA/nvidia-docker/issues/1447). This problems is confirmed to be in the process of fixing for **future releases**.

current fixes:

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


<b>Serving with BentoML</b>

after packing, edit Dockerfile as follows for GPU-supports:

```dockerfile

FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04 as nvidia-cuda

...
COPY --from=nvidia-cuda /usr/local/cuda-11.0 /usr/local/cuda
COPY --from=nvidia-cuda /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

...
# this is for pytorch only
RUN python -m spacy download en_core_web_sm

```

### PyTorch

_relevant files can be found under `*_torch.py`_

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Tensorflow

_relevant files can be found under `*_tf.py`_

```python
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# or using `with` statement:
with tf.device("/GPU:0"):
    ...

```