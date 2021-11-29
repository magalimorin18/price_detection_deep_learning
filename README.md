# price_detection_deep_learning

## Data


## Dev

- Install the dependencies `make install-dev`
- Setup the pre-commit script: `pre-commit install`


## Price annotation

To perform the price detection on the image, we need a dataset of the boxes on the images of the prices tickets.
We are using [VoTT](https://github.com/microsoft/VoTT), an open source tool from microsoft, to annotate a few images.
The idea is to proceed using the following steps:
- Annotate a few images (boxes coordinates and size)
- Train a model to predict the boxes positions
- Check the predictions and add them to the dataset
- Loop back until the model is good enough