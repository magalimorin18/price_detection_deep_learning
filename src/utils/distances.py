"""Distance between price and product."""

import pandas as pd


def compute_product_positions(products: pd.DataFrame) -> pd.DataFrame:
    """Compute product positions (Middle lower position)."""
    products["pos_x"] = (products.x1 + products.x2) / 2
    products["pos_y"] = products.y2
    return products


def compute_price_positions(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute prices positions (Middle upper position)."""
    prices["pos_x"] = (prices.x1 + prices.x2) / 2
    prices["pos_y"] = prices.y1
    return prices


def distance_function(x1, y1, x2, y2):
    """Compute distance between two points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def find_closest_price(product: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Return the product with the closest price info."""
    pr_x, pr_y = product.pos_x, product.pos_y
    distances = prices.apply(
        lambda row: distance_function(pr_x, pr_y, row.pos_x, row.pos_y), axis=1
    )
    distances = distances.sort_values()
    closest_distance = distances.head(1)
    closest_price = prices.loc[closest_distance.index.item()]

    pi_x, pi_y = closest_price.pos_x.item(), closest_price.pos_y.item()

    product["price_x1"] = closest_price.x1
    product["price_y1"] = closest_price.y1
    product["price_x2"] = closest_price.x2
    product["price_y2"] = closest_price.y2

    product["price_x"] = pi_x
    product["price_y"] = pi_y
    product["price_id"] = closest_price.index[0]

    return product
