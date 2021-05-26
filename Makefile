.DEFAULT_GOAL=pipe
TENSORFLOW_DIR=tf
PYTORCH_DIR=pytorch
ONNX_DIR=onnx

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean:
	rm -rf deploy/*

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
	cd deploy/onnx_svc && docker build -t bento-onnx-gpu:latest .

onnx-d-r:
	# ldconfig -p | grep nvidia
	docker run --gpus all -p 50053:5000 bento-onnx-gpu:latest

.PHONY: torch-e2e
torch-e2e: torch-train torch-pipe torch-d-r ## e2e pipeline from training to production torch on BentoML

torch-train:
	cd $(PYTORCH_DIR) && python3 train.py

.PHONY: torch-pipe
torch-pipe: torch-pack torch-d ## our pytorch deployment pipeline with bentoml

torch-pack:
	cd $(PYTORCH_DIR) && python3 bento_packer.py

torch-d:
	cd deploy/torch_svc && docker build -t bento-torch-gpu:latest .

torch-d-r:
	# ldconfig -p | grep nvidia
	docker run --gpus all -p 50052:5000 bento-torch-gpu:latest

.PHONY: tf-e2e
tf-e2e: tf-train tf-pipe tf-d-r  ## e2e pipeline from training to production tf on BentoML

tf-train:
	cd $(TENSORFLOW_DIR) && python3 train.py

.PHONY: tf-pipe
tf-pipe: tf-pack tf-d ## our tensorflow deployment pipeline with bentoml

tf-pack:
	cd $(TENSORFLOW_DIR) && python3 bento_packer.py

tf-d:
	cd deploy/tf_svc && docker build -t bento-tf-gpu:latest .

tf-d-r:
	docker run --gpus all -p 50051:5000 bento-tf-gpu:latest
