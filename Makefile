.DEFAULT_GOAL: prep

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

pack: torch-pack tf-pack

torch-train:
	python train_torch.py

torch-pack:
	python packer_torch.py

tf-train:
	python train_tf.py

tf-pack:
	python packer_tf.py

tf-d:
	cd deploy/tensorflow_service && docker build -t bento-tf-gpu:latest .

tf-d-r:
	docker run --gpus all --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset --device /dev/nvidiactl -p 5000:5000 bento-tf-gpu:latest

#tf-d-i:
#	docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash
