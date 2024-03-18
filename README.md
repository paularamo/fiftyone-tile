# FiftyOne Timestamps

Tile your images to squares (e.g. 960x960 pixels) in FiftyOne without exporting and re-importing.
Tested with with bounding boxes (without orientation) only.

## Walkthrough
1. If `resize` is set: Resize image to given width and keep aspect ratio
2. Add space around the image to make it a multiple of tiles size and place image in a random within the new boundaries.
3. Make tiles with the given `tile_size` and transfer available detections to the tiles.
   - Overlap tiles by `padding` value (in pixels)
   - Omit labels at image's borders if the don't reach in the image by `threshold` value (in pixels)
4. If `save_empty` is set, tiles without detections will be kept, if not omited.

<img src="screenshot.png" width="280">

## Installation
[Make sure OpenCV is installed](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html)
```shell
fiftyone plugins download https://github.com/mmoollllee/fiftyone-timestamps/
```

## Python SDK

You can use the compute operators from the Python SDK!

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("existing-dataset")

make_tiles = foo.get_operator("@mmoollllee/tile/make_tiles")

make_tiles(
   dataset,
   output_dir="filepath/to/save/tiles", # Required
   destination="destination_dataset_name", # defaults to current dataset name with '_tiled' suffix
   labels_field="ground_truth", # which labels to transfer to the tiles (Default: ground_truth)
   resize=1200, # resize the image before tiling (default: None)
   tile_size=960, # (default: 960)
   padding=20, # Overlap tiles by given value (default: 32),
   threshold=5, # Omit labels at the edged if smaller than given value (default: 10),
   save_empty: # Keep tiles without labels (default: False)
)
```
