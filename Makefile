lint:
	python -m pylint src
	python -m flake8 src


install-dev: install
	pip install -r requirements-dev.txt

install:
	pip install -r requirements.txt
