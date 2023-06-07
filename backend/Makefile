BASE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_DIR = $(BASE_DIR)/backend
VENV = $(BASE_DIR)/venv
# Replace with your python dir e.g. /usr/local/bin
PYTHON_DIR ?= /home/mzenkov/.local/bin
CONFIG_FILE ?= pyproject.toml

lint:
	$(PYTHON_DIR)/black --check $(SRC_DIR)
	$(PYTHON_DIR)/pflake8 --jobs 4 --statistics --show-source $(SRC_DIR)
	$(PYTHON_DIR)/pylint --jobs 4 --rcfile=$(CONFIG_FILE) $(SRC_DIR)
	$(PYTHON_DIR)/mypy --config-file=$(CONFIG_FILE) $(SRC_DIR)
	$(PYTHON_DIR)/bandit -r -ll --exclude $(VENV) $(SRC_DIR)

format:
	$(PYTHON_DIR)/isort $(SRC_DIR)
	$(PYTHON_DIR)/black $(SRC_DIR)
	$(PYTHON_DIR)/pautoflake $(SRC_DIR)
	$(PYTHON_DIR)/unify --in-place --recursive --quote "'" $(SRC_DIR)