## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
MAKEFLAGS += --silent
.PHONY: *
.ONESHELL:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = narcolepsy-detector
PYTHON_INTERPRETER = python
PLATFORM := $(shell uname)
PYTHON_VERSION = 3.10
CUDA_VERSION = 11.2
CUDNN_VERSION = 8.1
TF_VERSION = 2.14.0

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up Python interpreter environment
create-environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install package and dependencies with CPU version of TensorFlow
install-cpu: create-environment install-tensorflow-cpu install-package

## Install package and dependencies with GPU version of TensorFlow
ifeq ($(PLATFORM), Linux)
install-gpu: create-environment install-tensorflow-gpu install-package
else ifeq ($(PLATFORM), Darwin)
install-gpu: create-environment install-tensorflow-cpu install-package
endif

## Helper command for installing package
install-package:
	echo "----------------------------------------------------------------------"
	echo "Installing project package and requirements"
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) -m pip install -e .

## Helper command for installing CPU version of TensorFlow
install-tensorflow-cpu:
	echo "----------------------------------------------------------------------"
	echo "Installing TensorFlow CPU version"
ifeq ($(PLATFORM), Linux)
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) -m pip install tensorflow==$(TF_VERSION)
endif
ifeq ($(PLATFORM), Darwin)
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) -m pip install tensorflow-macos==$(TF_VERSION)
endif

## Helper command for installing GPU version of TensorFlow
install-tensorflow-gpu:
ifeq ($(PLATFORM), Darwin)
	echo "----------------------------------------------------------------------"
	echo "GPU installation is not available for MacOS, please run `make install-cpu` instead!"
else ifeq ($(PLATFORM), Linux)
	echo "----------------------------------------------------------------------"
	echo "Installing TensorFlow GPU version"
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) -m pip install tensorflow[and-cuda]==$(TF_VERSION)
else
	echo "----------------------------------------------------------------------"
	echo "Unsupported OS detected, cannot install package!"
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
