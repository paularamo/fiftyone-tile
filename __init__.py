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


def save_tile(padded_image, run, y, x, tile_size, padding, name, id, ext, output_dir):
    tiled = padded_image[
            y*tile_size:(y+1)*tile_size+padding, #0 : 960
            x*tile_size:(x+1)*tile_size+padding #0 : 960
        ]
    tile_name = name.replace(ext, f'_{id}_{run}_{y}_{x}{ext}')
    tile_path = os.path.join(output_dir, tile_name)
    cv2.imwrite(tile_path, tiled)
    return tile_path


def tile(image, tags, name, id, detections, output_dir, target: Dataset, save_empty=False, tile_size=960, padding=0, threshold=0.15, labels_field="ground_truth", runs=1):

    ext = os.path.splitext(name)[-1]
    height, width, channels = image.shape
    tile_size = tile_size - padding #928px

    temp = []
    for detection in detections:
        x, y, w, h = detection.bounding_box
        temp.append([detection, x*width, y*height, w*width, h*height])
    detections = pd.DataFrame(temp, columns=['detection','x1','y1','w','h'])

    # Calculate the number of tiles needed in each dimension
    num_tiles_x = math.ceil(width / tile_size) #3
    num_tiles_y = math.ceil(height / tile_size) #3

    # Pad the image to ensure it's divisible by the tile size
    padded_width = tile_size * num_tiles_x + padding #2.816px
    padded_height = tile_size * num_tiles_y + padding #2.816px

    # color for padding / background
    color = (144,144,144)

    runs_count = range(runs)
    runs = []

    for run in runs_count:
        # prepare run's data
        runs.append({
            "intersections": [],
            "samples": [],
            "below_threshold": 0,
            "detections_count": 0
        })

        # Randomize position on padded area
        padded_x = random.randint(0, padded_width - width) # 0 - 316px
        padded_y = random.randint(0, padded_height - height) # 0 - 942px
        padded_image = np.full((padded_height,padded_width, channels), color, dtype=np.uint8)
        padded_image[padded_y:padded_y+height, padded_x:padded_x+width] = image
        
        # rescale coordinates with padding
        temp_detections = detections.copy()
        temp_detections[['x1']] = detections[['x1']] + padded_x
        temp_detections[['y1']] = detections[['y1']] + padded_y

        # convert bounding boxes to shapely polygons.
        boxes = []
        for row in temp_detections.iterrows():
            i = row[1]
            x1 = i['x1']
            x2 = i['x1'] + i['w']
            y1 = (padded_height - i['y1']) - i['h']
            y2 = padded_height - i['y1']
            boxes.append((i['detection'], Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Coordinates of the tile
                tile_x1 = x*tile_size #0, 928, 1856
                tile_x2 = ((x+1)*tile_size) + padding #960, 1888, 2.816,...
                tile_y1 = padded_height - (y*tile_size) #2.816, 1.888, 960
                tile_y2 = (padded_height - (y+1)*tile_size) - padding #1856, 928, 0
                tile_box = Polygon([(tile_x1, tile_y1), (tile_x2, tile_y1), (tile_x2, tile_y2), (tile_x1, tile_y2)])

                imsaved = False
                tile_detections = []
                tile_intersections = []

                for box in boxes:
                    bbox = box[1]
                    if tile_box.intersects(bbox):
                        inter = tile_box.intersection(bbox)
                        intersection = inter.area / bbox.area

                        # Remove box if intersection is below threshold
                        if intersection < threshold:
                            runs[run]["below_threshold"] += 1
                            continue

                        # hacky copy of detection with all attributes, tags,...
                        detection = box[0].fancy_repr(class_name=None,exclude_fields=['id'])[12:-1]
                        detection = eval(detection)
                        detection = fo.Detection(**detection)

                        # add informations about if the detection was intersecting with the tiles boundaries
                        tile_intersections.append(intersection)
                        detection.set_field("intersection", intersection)
                        if intersection < 1:
                            detection.tags.append("intersecting")
                        
                        if not imsaved:
                            tile_path = save_tile(padded_image, run, y, x, tile_size, padding, name, id, ext, output_dir)
                            imsaved = True
                        
                        # get smallest rectangular polygon  that contains the intersection
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
                        
                        runs[run]["detections_count"] += 1

                        detection.bounding_box = [new_x, new_y, new_width, new_height]

                        tile_detections.append(detection)
                
                if not imsaved and save_empty:
                    tile_path = save_tile(padded_image, run, y, x, tile_size, padding, name, id, ext, output_dir)
                    imsaved = True

                if imsaved:
                    runs[run]["intersections"].extend(tile_intersections)

                    sample = fo.Sample(filepath=tile_path)
                    if len(tile_detections) > 0:
                        sample[labels_field] = fo.Detections(
                            detections=tile_detections
                        )
                    for tag in tags:
                        sample.tags.append(tag)

                    sample["intersection"] = sum(tile_intersections) / len(tile_intersections)

                    runs[run]["samples"].append(sample)

        runs[run]["intersections"] = sum(runs[run]["intersections"]) / len(runs[run]["intersections"])
    
    best_run = max(range(len(runs)), key=lambda index: runs[index]['intersections'])
    best_run = runs.pop(best_run)

    # add best samples
    for sample in best_run["samples"]:
        target.add_sample(sample)
    
    # delete the rest
    for run in runs:
        for sample in run["samples"]:
            os.remove(sample.filepath)

    return {
        "below_threshold": best_run["below_threshold"],
        "tiles_created": len(best_run["samples"]),
        "intersection": best_run['intersections']
        }


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

        log_level = ctx.params.get("log_level")

        output_dir = ctx.params.get("output_dir")
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if log_level >= 3:
                print(f"\tOutput directory created: " + output_dir)

        test = ctx.params.get("test")

        destination = ctx.params.get("destination") if ctx.params.get("destination") else ctx.dataset.name + "_tiled"
        try:
            tiled_dataset = fo.load_dataset(destination)
            if log_level >= 2:
                print(f"\tDestination dataset loaded: {destination}")
        except BaseException:
            tiled_dataset = fo.Dataset(destination)
            tiled_dataset.persistent = False if test else True
            if log_level >= 2:
                print(f"\tDestination dataset created (persistent={tiled_dataset.persistent}): {destination}")

        save_empty = ctx.params.get("save_empty")

        labels_field = ctx.params.get("labels_field")
        if not save_empty:
            if labels_field:
                view = view.match(F(labels_field+".detections").length())
            else:
                view = []

        if test:
            print(" 🤪\tTestmode!")
            view = view.head(5)

        resize = ctx.params.get("resize")
        tile_size = ctx.params.get("tile_size")
        padding = ctx.params.get("padding")
        threshold = ctx.params.get("threshold")
        runs = ctx.params.get("runs")
        name = ctx.params.get("name")

        if not name:
            name = view.dataset if "dataset" in view else view.name
        if log_level >= 1:
            print(f" 🚀\tStarting with {len(view)} samples in {name}")

        stats = {
            "below_threshold": 0,
            "tiles_created": 0,
            "intersections": []
        }

        for sample in view:
            filepath = sample['filepath']
            if log_level >= 3:
                print(" 🌆\t"+filepath)

            if labels_field and not save_empty and not sample[labels_field]:
                if log_level >= 2:
                    print(f"  🧐\tNo {labels_field} detections found: Skiping")
                continue
            
            image = cv2.imread(filepath)

            if resize:
                image = image_resize(image, width=resize)
            
            if labels_field and sample[labels_field]:
                labels = sample[labels_field].detections
            else:
                if log_level >= 3:
                    print(f"  🧐\tNo {labels_field} detections found")
                labels = None

            stat = tile(image, sample.tags, os.path.basename(filepath), sample['id'], labels, output_dir, target=tiled_dataset, tile_size=tile_size, padding=padding, threshold=threshold, save_empty=save_empty, labels_field=labels_field, runs=runs)

            stats["below_threshold"] += stat["below_threshold"]
            stats["tiles_created"] += stat["tiles_created"]
            stats["intersections"].append(stat["intersection"])
            if log_level >= 2:
                print(f"  🪚\t{str(stat['tiles_created'])} tiles\t\t⌀ intersection {str(round(stat['intersection'],2))}\t{str(stat['below_threshold'])} detections < threshold. \t{os.path.basename(filepath)}")

        samples_total = len(view)
        tiles_per_sample = str(round(stats["tiles_created"]/samples_total, 1))
        below_threshold_per_sample = str(round(stats["below_threshold"]/samples_total, 1))
        tiles_created = str(stats['tiles_created'])
        below_threshold = str(stats["below_threshold"])
        avg_intersection = str(round(sum(stats["intersections"]) / samples_total, 4))

        print(f" ✅\t{name} done with {samples_total} samples")
        print(f"\tCreated {tiles_per_sample} tiles/sample (total: {tiles_created})")
        print(f"\tLost {below_threshold_per_sample} detections/sample below threshold({str(threshold)}) (total: {below_threshold})")
        print(f"\tAverage intersection {avg_intersection}")

    def __call__(
        self, 
        sample_collection, 
        output_dir: str, 
        name: Optional[str] = None,
        destination: Optional[str] = None,
        labels_field: Optional[str] = "ground_truth",
        resize: Optional[int] = None,
        tile_size: int = 960,
        padding: int = 32,
        threshold: float = 0.15,
        save_empty: bool = False,
        test: bool = False,
        runs: int = 1,
        log_level: int = 2
    ):
        ctx = dict(view=sample_collection.view())
        params = dict(
            target="CURRENT_VIEW",
            output_dir=output_dir,
            name=name,
            destination=destination,
            labels_field=labels_field,
            resize=resize,
            tile_size=tile_size,
            padding=padding,
            threshold=threshold,
            save_empty=save_empty,
            test=test,
            runs=runs,
            log_level=log_level
            )
        return foo.execute_operator(self.uri, ctx, params=params)


def register(p):
    p.register(MakeTiles)
