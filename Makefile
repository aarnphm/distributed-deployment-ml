.PHONY := all

.DEFAULT_GOAL := build

all: help build

DEPLOY_DIR := deploy
SVC_DIR := $(HOME)/bentoml/repository/ProfanityFilterService
DOT_DATA := $(DEPLOY_DIR)/ProfanityFilterService/.data

# ls -ltr $(SVC_DIR) | grep '^d' | tail -1 | awk '{print $(NF)}'
LATEST := $(shell ls -t -- $(SVC_DIR)/* | head -n1 | cut -d ':' -f1)

help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'	

prep: ## Prepare embedding, using GloVe
	cd src && python train.py $(ARGS)

init: prep ## package trained model to bento
	cd src && python packer.py

build: ## build bento docker images then deploy on 5000
	rm -rf $(DEPLOY_DIR) && cp -r $(LATEST) $(DEPLOY_DIR) && mkdir $(DOT_DATA)
	cp -r src/.data/{aclImdb,aclImdb_v1.tar.gz,glove.6B.50d.txt}  $(DOT_DATA)	
	cp src/requirements.txt $(DEPLOY_DIR) && cp src/config.yml $(DEPLOY_DIR)/ProfanityFilterService
	echo 'pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html' >> $(DEPLOY_DIR)/bentoml-init.sh
	cd $(DEPLOY_DIR) && docker build -t profanity-filter:latest .
	
run: ## run docker images
	docker run -p 5000:5000 profanity-filter:latest
