# price_detection_deep_learning

## Data

The structure of the data folder is the following:
```
- data
    - train
        - images
            - 0000.jpg
            - 0001.jpg
        - annocations.csv
    - test
        - images
            - 0000.jpg
            - 0001.jpg
        - results.csv
```

You can download the data from https://www.kaggle.com/itamargr/traxpricing-dataset, it is from the pricing challenge 2021 of Retail Vision for CVPR 2021 workshop https://retailvisionworkshop.github.io/pricing_challenge_2021/.


## Dev

- Install the dependencies `make install-dev`


## Price annotation

To perform the price detection on the image, we need a dataset of the boxes on the images of the prices tickets.
We are using [VoTT](https://github.com/microsoft/VoTT), an open source tool from microsoft, to annotate a few images.
The idea is to proceed using the following steps:
- Annotate a few images (boxes coordinates and size)
- Train a model to predict the boxes positions
- Check the predictions and add them to the dataset
- Loop back until the model is good enough
