"""VoTT data loader."""


import pandas as pd


def load_vott_data(path: str) -> pd.DataFrame:
    """Load VoTT data (csv) and format.

    :params path: The path of the csv file,
    with the columns: image,xmin,ymin,xmax,ymax,label

    :returns: A dataframe containing the annotations,
    with the columns: image,x1,x2,y1,y2
    """
    df = pd.read_csv(path).drop(columns=["label"])
    df = df.rename(
        columns={
            "xmin": "x1",
            "xmax": "x2",
            "ymin": "y1",
            "ymax": "y2",
            "image": "img_name",
        }
    )
    return df


if __name__ == "__main__":
    load_vott_data("./data/vott-csv-export/Price-detection-export.csv")
