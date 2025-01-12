dependencies: ## Install all project requirements
	sudo apt update && sudo apt install ffmpeg -y
	pip install -r requirements.txt

.PHONY: data

train:
	@python3 trainSTFT.py 
	@cp metrics.csv metrics_stft.csv
	@python3 trainMEL.py 
	@cp metrics.csv metrics_mel.csv
	@python3 trainDOL.py 
	@cp metrics.csv metrics_dol.csv

attack:
	@python3 benchmark_attacks.py

defence:
	@python3 defence/classify_detect.py
	@python3 defence/compare_detect.py

data: ## Fetch the data as a zip file
	@mkdir -p data
	@curl -L -o ./data/archive.zip https://www.kaggle.com/api/v1/datasets/download/tli725/jl-corpus
	@unzip data/archive.zip -d data/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'