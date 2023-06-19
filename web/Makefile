SRC_FILE = web/main.py
# Replace with your python dir e.g. /home/mzenkov/.local/bin
PYTHON_DIR ?= /home/runner/.local/bin
CONFIG_FILE ?= pyproject.toml

lint:
	$(PYTHON_DIR)/black --check $(SRC_FILE)
	$(PYTHON_DIR)/pflake8 --jobs 4 --statistics --show-source $(SRC_FILE)
	$(PYTHON_DIR)/pylint --jobs 4 --rcfile=$(CONFIG_FILE) $(SRC_FILE)
	$(PYTHON_DIR)/mypy --config-file=$(CONFIG_FILE) $(SRC_FILE)
	$(PYTHON_DIR)/bandit -r -ll $(SRC_FILE)

format:
	$(PYTHON_DIR)/isort $(SRC_FILE)
	$(PYTHON_DIR)/black $(SRC_FILE)
	$(PYTHON_DIR)/pautoflake $(SRC_FILE)
	$(PYTHON_DIR)/unify --in-place --recursive --quote "'" $(SRC_FILE)
