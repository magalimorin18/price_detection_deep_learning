lint:
	python -m pylint src notebooks
	python -m flake8 src notebooks


install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

install:
	pip install wheel
	pip install -r requirements.txt


