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

tf-d-i:
	docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash
