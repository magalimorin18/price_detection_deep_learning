"""Remove the overlap of price tags over products."""

import pandas as pd


# pylint: disable=too-many-arguments
def overlap_area(px1, py1, px2, py2, qx1, qy1, qx2, qy2, bottle_factor=0.15):
    """Bottle first, then price tag."""
    # We reduce the area of the bounding box of the bottle
    # To avoid the case where the bottle is in the price tag
    px1 = px1 + (px2 - px1) * bottle_factor
    py1 = py1 + (py2 - py1) * bottle_factor
    px2 = px2 - (px2 - px1) * bottle_factor
    py2 = py2 - (py2 - py1) * bottle_factor

    x1 = max(px1, qx1)
    y1 = max(py1, qy1)
    x2 = min(px2, qx2)
    y2 = min(py2, qy2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def remove_overlaping_tags(products: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Remove the overlaping tags."""
    to_remove = set()
    for q in prices.itertuples():
        if q.Index in to_remove:
            break
        for p in products.itertuples():
            overlap_score = overlap_area(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1, q.x2, q.y2)
            # product_area = (p.x2 - p.x1) * (p.y2 - p.y1)
            price_area = (q.x2 - q.x1) * (q.y2 - q.y1)
            score = overlap_score / (price_area + 1e-10)
            if score > 0.5:
                to_remove.add(q.Index)
                break
    return prices.drop(list(to_remove))
