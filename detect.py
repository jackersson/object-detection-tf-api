"""
Usage:

>>> Detects object on whole image
    python detect.py -s data/images/normal.jpg -c configs/tf_object_api_cfg.yml -p "[0,0], [0,1], [1,1], [1,0]"
"""
import os
import argparse
import itertools
import typing as typ

import cv2
import numpy as np

from shapely.geometry import box as Box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from utils import TfObjectDetectionModel


FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.5
TEXT_THICKNESS = 1
LINE_THIKNESS = 3
TEXT_COLOR = (255, 255, 255)
LINE_COLOR = (255, 0, 0)
POLYGON_COLOR = (60, 179, 113)
POLYGON_ALPHA = 0.3
MASK_ALPHA = 0.5

LINE_IN_COLOR = (0, 155, 0)
LINE_OUT_COLOR = (255, 0, 0)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source", required=True,
                        help="Path to image file")
    parser.add_argument("-c", "--config", required=False, default="config.yaml",
                        help="Path to configuration file (TF Inference)")
    parser.add_argument("-o", "--out", required=False, default="out.jpg",
                        help="Path to output image file")
    parser.add_argument("-p", "--polygon", required=False, default="[0, 0], [0, 1], [1, 1], [1, 0]",
                        help="Specify target area (polygon). Format: list of points [x, y], .. (relative coords)")

    parser.add_argument("-a", "--area_threshold", required=False, default=1e-3,
                        help="Fraction of area to be occupied by object. Float [0.0, 1.0] ")

    return vars(parser.parse_args())


def draw_detections(image: np.ndarray, detections: typ.List[dict],
                    color: typ.Tuple[int, int, int] = LINE_COLOR, mask_alpha: float = MASK_ALPHA,
                    text_color: typ.Tuple[int, int, int] = TEXT_COLOR) -> np.ndarray:
    """Draws bounding boxes, masks and class name over an image"""

    mask_alpha = np.clip(mask_alpha, 0, 1.0)
    for detection in detections:
        left, top, width, height = detection['bounding_box']
        right, bottom = left + width, top + height

        # draw mask
        if 'mask' in detection:
            mask = detection['mask']
            roi = image[top:bottom, left:right][mask]

            overlay = ((mask_alpha * np.array(color, dtype=np.uint8)) + ((1.0 - mask_alpha) * roi)).astype(np.uint8)
            image[top:bottom, left:right][mask] = overlay

        # draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=LINE_THIKNESS)

        # draw class_name
        text = detection['class_name']
        text_width, text_height = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)[0][:2]

        # draw tableu
        cv2.rectangle(image, (left, top - text_height), (right, top), color, cv2.FILLED)

        # put text
        cv2.putText(image, text, (left, top), FONT, FONT_SCALE, text_color, TEXT_THICKNESS)

    return image


def draw_polygon(image: np.ndarray, points: typ.List[typ.Tuple[int, int]],
                 alpha=POLYGON_ALPHA, color=POLYGON_COLOR) -> np.ndarray:
    """Draw polygon (closed area)"""
    image_copy = image.copy()
    cv2.fillPoly(image_copy, np.array([points]), color)

    cv2.addWeighted(image_copy, alpha, image, 1-alpha, 0, image)
    return image


def is_inside(a: BaseGeometry, b: BaseGeometry, *, threshold: float = 0.05) -> bool:
    """ Checks if one geometry (a) is inside other (b)"""
    return a.intersection(b).area/(b.area + 1e-8) >= threshold


def to_box(box: typ.Tuple[int, int, int, int]) -> Box:
    """Converts box [top, left, right, bottom] to shapely.Box"""
    x, y, w, h = box
    return Box(x, y, x + w, y + h)


def main():
    parsed_args = parse_arguments()

    if not parsed_args:
        sys.exit(1)

    source = parsed_args['source']
    if not os.path.isfile(source):
        print(f"Invalid source: {source}")
        sys.exit(1)

    # loading image
    image = cv2.imread(source, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # parse target area (polygon)
    points = [float(pt) for pt in parsed_args['polygon'].replace('[', "").replace(']', "").split(',')]
    points = np.array(points).reshape(len(points) // 2, 2) * image.shape[:2][::-1]
    points = points.astype(np.int32)
    polygon = Polygon(points.tolist())  # Polygon

    # draw polygon
    out_image = draw_polygon(image.copy(), points.tolist())

    # do object detection
    with TfObjectDetectionModel.from_config_file(parsed_args['config']) as model:
        detections = model.process_single(image)

        inside = [is_inside(to_box(det['bounding_box']), polygon, threshold=float(
            parsed_args['area_threshold'])) for det in detections]

        draw_detections(out_image, itertools.compress(
            detections, inside), color=LINE_IN_COLOR)

        draw_detections(out_image, itertools.compress(
            detections, [not v for v in inside]), color=LINE_OUT_COLOR)

    # save to file
    _, ext = os.path.splitext(os.path.basename(source))
    out_dirname = os.path.dirname(os.path.abspath(parsed_args['out']))
    os.makedirs(out_dirname, exist_ok=True)
    out_filename, _ = os.path.splitext(os.path.basename(source))
    out_filename = os.path.join(out_dirname, "det_{}{}".format(out_filename, ext))

    cv2.imwrite(out_filename, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
    print(f"Saved to {out_filename}")


if __name__ == '__main__':
    main()
