# Deep Learning Project -

[![wakatime](https://wakatime.com/badge/user/5fba56dd-c3e1-4bec-9596-fd1565702df9/project/a1c8fad7-d886-4e60-bd73-1a94648fa163.svg)](https://wakatime.com/badge/user/5fba56dd-c3e1-4bec-9596-fd1565702df9/project/a1c8fad7-d886-4e60-bd73-1a94648fa163)
[![CI](https://github.com/magalimorin18/price_detection_deep_learning/actions/workflows/main.yml/badge.svg)](https://github.com/magalimorin18/price_detection_deep_learning/actions/workflows/main.yml)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/magalimorin18/price_detection_deep_learning)
![GitHub last commit](https://img.shields.io/github/last-commit/magalimorin18/price_detection_deep_learning)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/magalimorin18/price_detection_deep_learning)

This project was created to practice deep learning on a dataset of prices of products. The dataset comes from the pricing challenge 2021 of Retail Vision for CVPR 2021 workshop https://retailvisionworkshop.github.io/pricing_challenge_2021/.
The purpose of the final model is to identify the prices of each products on an image in a supermarket for instance.

You can check our report here: https://www.overleaf.com/read/vndycspxvfxr

## How to use the code

- Install the dependencies `make install-dev`
- Install the torch dependencies
    - On CPU: `make install-cpu`
    - On GPU: `make install-gpu`
- __You can download the data and model weights form the following Google Drive folder: https://drive.google.com/drive/folders/147fTbdhXe5UB6iGQicHKYLORPxx6wf4_?usp=sharing__
- Run the streamlit interface `make run`

## Price annotation

To perform the price detection on the image, we need a dataset of the boxes on the images of the prices tickets.
We are using [jupyter_bbox_widget](https://github.com/gereleth/jupyter-bbox-widget), a module that allows to annotate images directly in a jupyter notebook, and is easy to use + easy to retrieve annotations and put new annotations from our model.
The idea is to proceed using the following steps:
- Annotate a few images (boxes coordinates and size)
- Train a model to predict the boxes positions
- Check the predictions and add them to the dataset
- Loop back until the model is good enough
