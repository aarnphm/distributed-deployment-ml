.DEFAULT_GOAL=pipe
TENSORFLOW_DIR=tf
PYTORCH_DIR=pytorch
ONNX_DIR=onnx

# handles cgroup v2 changes and nvidia-docker caveats
ifeq (,$(wildcard, /etc/arch-release))
	DEVICE_ARGS := --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
endif

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

compose: ## run all services together
	docker-compose up

.PHONY: pipe
pipe: torch-pipe tf-pipe onnx-pipe ## run tensorflow and torch, and onnx pipe

.PHONY: onnx-e2e
onnx-e2e: onnx-pipe onnx-d-r ## e2e pipeline from training to production onnx on BentoML

.PHONY: onnx-pipe
onnx-pipe: onnx-pack onnx-d ## our ONNX deployment pipeline with bentoml

onnx-pack:
	cd $(ONNX_DIR) && CUDA_LAUNCH_BLOCKING=1 python3 bento_packer.py

onnx-d:
	bentoml containerize OnnxService:latest -t bentoml-onnx-gpu:latest --verbose

onnx-d-r:
	docker run --gpus all -p 50000:5000 $(DEVICE_ARGS) bentoml-onnx-gpu:latest

.PHONY: torch-e2e
torch-e2e: torch-train torch-pipe torch-d-r ## e2e pipeline from training to production torch on BentoML

torch-train:
	cd $(PYTORCH_DIR) && python3 train.py

.PHONY: torch-pipe
torch-pipe: torch-pack torch-d ## our pytorch deployment pipeline with bentoml

torch-pack:
	cd $(PYTORCH_DIR) && python3 bento_packer.py

torch-d:
	bentoml containerize PytorchService:latest -t bentoml-torch-gpu:latest --verbose

torch-d-r:
	docker run --gpus all -p 50000:5000 $(DEVICE_ARGS) bentoml-torch-gpu:latest

.PHONY: tf-e2e
tf-e2e: tf-train tf-pipe tf-d-r  ## e2e pipeline from training to production tf on BentoML

tf-train:
	cd $(TENSORFLOW_DIR) && python3 train.py

.PHONY: tf-pipe
tf-pipe: tf-pack tf-d ## our tensorflow bentoml_servicement pipeline with bentoml

tf-pack:
	cd $(TENSORFLOW_DIR) && python3 bento_packer.py

tf-d:
	bentoml containerize TensorflowService:latest -t bentoml-tensorflow-gpu:latest --verbose

tf-d-r:
	docker run --gpus all -p 50000:5000 $(DEVICE_ARGS) bentoml-tensorflow-gpu:latest
