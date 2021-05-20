.DEFAULT_GOAL=pipe

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean:
	rm -rf deploy/*

.PHONY: pipe
pipe: tf-pipe torch-pipe ## init both pipeline

.PHONY: torch-pipe
torch-pipe: torch-train torch-pack torch-d ## our torch pipeline with bentoml

torch-train:
	python train_torch.py

torch-pack:
	# RUN python -m spacy download en_core_web_sm for Docker
	python packer_torch.py

torch-d:
	cd deploy/pytorch_service && docker build -t bento-torch-gpu:latest .

torch-d-r:
	docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl -p 5000:5000 bento-torch-gpu:latest

.PHONY: tf-pipe
tf-pipe: tf-train tf-pack tf-d ## our tensorflow pipeline with bentoml

tf-train:
	python train_tf.py

tf-pack:
	python packer_tf.py

tf-d:
	cd deploy/tensorflow_service && docker build -t bento-tf-gpu:latest .

tf-d-r:
	docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl -p 5000:5000 bento-tf-gpu:latest