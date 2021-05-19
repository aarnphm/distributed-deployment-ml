.DEFAULT_GOAL: prep

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'	

train-torch:
	python train.py --train

train-tf:
	python train.py --tf

train-prep: ## prep embedding
	python train.py

.PHONY: dist-run ## run server with distributed
dist-run:
	python -m torch.distributed.launch --nproc_per_node=2 server.py

.PHONY: run ## run our server
run:
	python server.py

.PHONY: guni ## start production
guni:
	gunicorn --bind 0.0.0.0:5000 server:app --log-level=debug -t 120 --workers=4

.PHONY: sentiment ## run sentiment analysis
sentiment:
	curl -X POST localhost:5000/api/sentiment --form 'text=When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.'

.PHONY: poetry-torch
poetry-torch:
	if [[ ! -f ./venv/torch.whl ]]; then \
		curl -o ./venv/torch.whl https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl; \
	fi
	poetry add ./venv/torch.whl
