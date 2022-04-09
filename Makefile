lint:
	python -m pylint src notebooks
	python -m flake8 src notebooks
mypy:
	python -m mypy src

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

install:
	pip install wheel
	pip install -r requirements.txt

install-cpu: install
	pip3 install torch==1.10.0 torchvision==0.11.1

install-gpu: install
	pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

run:
	streamlit run ./streamlit_app.py