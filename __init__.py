import os
import cv2
from typing import Optional
import pandas as pd
import math
import random
import numpy as np
from shapely.geometry import Polygon

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone import ViewField as F
from fiftyone import Dataset


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def save_tile(padded_image, y, x, tile_size, padding, name, ext, output_dir):
    tiled = padded_image[
            y*tile_size:(y+1)*tile_size+padding, #0 : 960
            x*tile_size:(x+1)*tile_size+padding #0 : 960
        ]
    tile_name = name.replace(ext, f'_tiled_{y}_{x}{ext}')
    tile_path = os.path.join(output_dir, tile_name)
    cv2.imwrite(tile_path, tiled)
    return tile_path

def tile(image, name, detections, output_dir, target: Dataset, save_empty=False, tile_size=960, padding=0, threshold=0, labels_field="ground_truth"):
    ext = os.path.splitext(name)[-1]
    height, width, channels = image.shape
    tile_size = tile_size - padding #920px

    df_list = []
    for detection in detections:
        x, y, w, h = detection.bounding_box
        df_list.append([detection.label, x*width, y*height, w*width, h*height])
    labels = pd.DataFrame(df_list, columns=['label','x1','y1','w','h'])

    # Calculate the number of tiles needed in each dimension
    num_tiles_x = math.ceil(width / tile_size) #5
    num_tiles_y = math.ceil(height / tile_size) #4

    # Pad the image to ensure it's divisible by the tile size
    padded_width = tile_size * num_tiles_x + padding #4.640px
    padded_height = tile_size * num_tiles_y + padding #3.720px

    # Randomize position on padded area
    padded_x = random.randint(0, padded_width - width) # 0 - 480px
    padded_y = random.randint(0, padded_height - height) # 0 - 600px

    color = (144,144,144)
    padded_image = np.full((padded_height,padded_width, channels), color, dtype=np.uint8)
    padded_image[padded_y:padded_y+height, 
       padded_x:padded_x+width] = image
    
    # rescale coordinates with padding
    labels[['x1']] = labels[['x1']] + padded_x
    labels[['y1']] = labels[['y1']] + padded_y

    # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
    boxes = []
    for row in labels.iterrows():
        i = row[1]
        x1 = i['x1']
        x2 = i['x1'] + i['w']
        y1 = (padded_height - i['y1']) - i['h']
        y2 = padded_height - i['y1']
        boxes.append((i['label'], Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

    counter = 0
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Coordinates of the tile
            tile_x1 = x*tile_size #0, 920, 1840,...
            tile_x2 = ((x+1)*tile_size) + padding #959, 1879, 2.799,...
            tile_y1 = padded_height - (y*tile_size) #3.720, 2.800, 1.880,...
            tile_y2 = (padded_height - (y+1)*tile_size) - padding #2.761, 1.1841, 921,...
            tile_pol = Polygon([(tile_x1, tile_y1), (tile_x2, tile_y1), (tile_x2, tile_y2), (tile_x1, tile_y2)])

            # Ignore bounding boxes by defined threshold
            boxes_x1 = tile_x1 + threshold
            boxes_x2 = tile_x2 - threshold
            boxes_y1 = tile_y1 - threshold
            boxes_y2 = tile_y2 + threshold
            boxes_pol = Polygon([(boxes_x1, boxes_y1), (boxes_x2, boxes_y1), (boxes_x2, boxes_y2), (boxes_x1, boxes_y2)])

            imsaved = False
            tile_labels = []

            for box in boxes:
                if boxes_pol.intersects(box[1]):
                    inter = tile_pol.intersection(box[1])
                    
                    if not imsaved:
                        tile_path = save_tile(padded_image, y, x, tile_size, padding, name, ext, output_dir)
                        imsaved = True
                    
                    # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                    new_box = inter.envelope 
                    
                    # get central point for the new bounding box 
                    centre = new_box.centroid
                    
                    # get coordinates of polygon vertices
                    box_x, box_y = new_box.exterior.coords.xy
                    
                    # get bounding box width and height normalized to tile size
                    new_width = (max(box_x) - min(box_x)) / (tile_size+padding)
                    new_height = (max(box_y) - min(box_y)) / (tile_size+padding)
                    
                    # we have to normalize central x and invert y for yolo format
                    new_x = (centre.coords.xy[0][0] - tile_x1) / (tile_size+padding) - (new_width/2)
                    new_y = (tile_y1 - centre.coords.xy[1][0]) / (tile_size+padding) - (new_height/2)
                    
                    counter += 1

                    tile_labels.append(fo.Detection(label=box[0], bounding_box=[new_x, new_y, new_width, new_height]))
            
            if not imsaved and save_empty:
                tile_path = save_tile(padded_image, y, x, tile_size, padding, name, ext, output_dir)
                imsaved = True

            if imsaved:
                sample = fo.Sample(filepath=tile_path)
                if len(tile_labels) > 0:
                    sample[labels_field] = fo.Detections(
                        detections=tile_labels
                    )
                target.add_sample(sample)


################################################################
################################################################

class MakeTiles(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="make_tiles",
            label="Create Tiles from images",
            description="Tile your images to squares (e.g. 960x960 pixels) in FiftyOne without exporting and re-importing."
        )


    def execute(self, ctx):
        view = ctx.view
        if view is None:
            view = ctx.dataset

        output_dir = ctx.params.get("output_dir")
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        destination = ctx.params.get("destination") if ctx.params.get("destination") else ctx.dataset.name + "_tiled"
        try:
            tiled_dataset = fo.load_dataset(destination)
            print(f"Destination dataset loaded: {destination}")
        except BaseException:
            tiled_dataset = fo.Dataset(destination)
            tiled_dataset.persistent = True
            print(f"Destination dataset created (persistent): {destination}")

        save_empty = ctx.params.get("save_empty")

        labels_field = ctx.params.get("labels_field")
        if not save_empty:
            if labels_field:
                view.match(F(labels_field+".detections").length())
            else:
                view = []

        resize = ctx.params.get("resize")
        tile_size = ctx.params.get("tile_size")
        padding = ctx.params.get("padding")
        threshold = ctx.params.get("threshold")

        print(f"Samples found: {len(view)}")

        for sample in view:
            filepath = sample['filepath']
            print(filepath)

            if labels_field and not save_empty and not sample.has_field(labels_field):
                print(f"  No {labels_field} detections found: Skiping")
                continue
            
            image = cv2.imread(filepath)

            if resize:
                image = image_resize(image, width=resize)
            
            if labels_field and sample.has_field(labels_field):
                print("  Tiling...")
                labels = sample[labels_field].detections
            else:
                print(f"  No {labels_field} detections found: Tiling anyway.")
                labels = None

            tile(image, os.path.basename(filepath), labels, output_dir, target=tiled_dataset, tile_size=tile_size, padding=padding, threshold=threshold, save_empty=save_empty, labels_field=labels_field)

    def __call__(
        self, 
        sample_collection, 
        output_dir: str, 
        destination: Optional[str] = None,
        labels_field: Optional[str] = "ground_truth",
        resize: Optional[int] = None,
        tile_size: int = 960,
        padding: int = 32,
        threshold: int = 10,
        save_empty: bool = False
    ):
        ctx = dict(view=sample_collection.view())
        params = dict(
            target="CURRENT_VIEW",
            output_dir=output_dir,
            destination=destination,
            labels_field=labels_field,
            resize=resize,
            tile_size=tile_size,
            padding=padding,
            threshold=threshold,
            save_empty=save_empty
            )
        return foo.execute_operator(self.uri, ctx, params=params)


def register(p):
    p.register(MakeTiles)
