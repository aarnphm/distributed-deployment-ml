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
pipe: torch-pipe tf-pipe ## run both tensorflow and torch pipe

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
	docker run --gpus all -p 7000:5000 bento-torch-gpu:latest

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
	docker run --gpus all -p 6000:5000 bento-tf-gpu:latest