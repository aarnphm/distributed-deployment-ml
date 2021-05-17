.DEFAULT_GOAL: prep

.PHONY: help
help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'	

.PHONY: run
run:
	python -m torch.distributed.launch --nproc_per_node=1 server.py

.PHONY: guni
guni:
	gunicorn --bind 0.0.0.0:5000 server:app --log-level=debug -t 120 --workers=4

.PHONY: categorize
categorize:
	curl -X POST localhost:5000/api/categorize --form 'text=The root of the latest escalation was intense disputes over East Jerusalem. Israeli police prevented Palestinians from gathering near one of the city’s ancient gates during the holy month of Ramadan, as they had customarily. At the same time, Palestinians faced eviction by Jewish landlords from homes in East Jerusalem. Many Arabs called it part of a wider Israeli campaign to force Palestinians out of the city, describing it as ethnic cleansing.'

.PHONY: sentiment
sentiment:
	curl -X POST localhost:5000/api/sentiment --form 'text=When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.'

.PHONY: poetry-torch
poetry-torch:
	if [[ ! -f ./venv/torch.whl ]]; then \
		curl -o ./venv/torch.whl https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl; \
	fi
	poetry add ./venv/torch.whl
