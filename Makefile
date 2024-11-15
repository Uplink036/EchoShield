dependencies: ## Install all project requirements
	sudo apt update && sudo apt install ffmpeg -Y
	pip install -r requirements.txt

.PHONY: data
data:
	mkdir -p data
	curl -L -o ./data/archive.zip https://www.kaggle.com/api/v1/datasets/download/tli725/jl-corpus
	
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'