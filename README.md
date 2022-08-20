**Some information in this repository as well as this readme is intentionally
left out as of an anonymous review process.**

This repository contains ready-to-use configuration files for the dataset of
our paper (90 nm, 65 nm, 40 nm and 28 nm). These files can simply be run in a
shell and will output our research results in a convenient HTML document. This
allows for a direct reproducibility of the studies.

In case the replaced cell instances need to be known for further studies
without re-running all algorithms, please feel free to contact the authors via
email **REDACTED**.

Setup
=====

Tested on Python 3.10.5.

The required Python dependencies can be installed using `pip install -r requirements.txt`.
This will install the following modules:
```
gdspy
numpy
opencv
imutils
```

Having the datasets in the `data` directory, the `run_XXnm.py` files might be run directly from the shell, e.g.:
```
python run_28.py fill
```
or
```
python run_28.py std
```

There is also a demo data set (only filler cell detection) already included in this repository, which can be run with:
```
python run_90nm_demo.py fill
```

The resulting data is output in the `output` directory.

General Idea
============

The general idea of chipsuite was to develop a tool that can compare real chip
imagery against GDSII files, but it turned out to be more powerful. By now it
is possible to...

* load global stitching information from the MIST tool
* load GDSII bounding boxes into a list and decide whether the bounding box is
  a standard cell or a filler cell
* shift the design coordinates to the global image coordinates with a 4-edge
  affine transformation
* correct the transformed coordinates on a tile / coordinate base
* run various algorithms on the combination of tile / cell image and design
  bounding box
  * detect brighter areas and based on a threshold decide whether the cell
    image matches the bounding box
  * detect vias (circular structures of a set size) and decide based on the
    probability of vias whether the cell image matches the bounding box
  * decide whether the cell image matches a cell image of another cell image
    using template matching / correlation based on the bounding box

Modules
=======

This project consists out of seperate modules for different parts of the
functionality it is designed for.

`gds_loader.GDSLoader`
* Opens up a GDSII file and applies a 4-edge transformation on all bounding
  boxes inside.
* Stores the state of the calculated bounding boxes and supplementary
  information in a pickle file to be able to reload the state without
  the need to recalculate all the values and opening up large GDSII files
  again.

`stitching_info.StitchingInfo`
* Opens up MIST global position files.
* Analyzes minimum / maximum coordinates and count all tiles.
* Detects the tile size based on the first tile image's size.

`bbox_generator.BboxGenerator`
* Applies a user set correction function for X / Y coordinates each.
* Provides a generator to iterate through every bounding box on a specific
  tile.
* Helper functions to draw bounding boxes on tiles and to show this image.

`algorithm.Algorithm`
* Base functions available in all algorithms, such as iterating over every
  tile.

`algorithm.Algorithm2`
* Detects circular structures (vias).
* Detects repeating structure of filler cells. (experimental)

`algorithm.Algorithm3` (`3_2`, `3_3`, `3_4`, `3_5`)
* Utilizes template matching to detect different cells

`config.Config`
* Contains colors for displayed bounding boxes (colorblind-proof!).

`cvhelper`
* Contains some extra functions to be used together with OpenCV.

`powerline` (`1`, `2`)
* Two power line detection algorithms.

`identify.CellIdentifier`
* Based on template matching and a library of cell templates, find out which
  of the template matches best given a cell image.

How This Works
==============

Preparations:
* Stitch the tiles with MIST/BigStitcher/... . MIST outputs the correct format
  in this "global-positions" textfile that can directly be used by
  `StitchingInfo.load_mist_file(...)`.
* Check orientation difference between GDSII and stitched & fused image, apply
  parameter `extra_transformation` of `GDSLoader.load_gds(...)` appropriately.
* Review all four tiles that show edges of the chip and put the global
  coordinates of the edges into the `image_edges` list of
  `GDSLoader.load_gds(...)`.

Tune Parameters:
* Let the script run in interactive mode "`-i`".
* After the first tile is analyzed, stop the process (Ctrl+C), now you are in
  the python interactive console.
* Review how say every fifth tile with cells on it looks like, do this by
  typing for example `bbox_generator.show_boxes(5, 5)`. (Any of these image
  windows can only be closed using any keyboard key for now, every other
  approach might crash the script.)
  * The coordinate offsets between GDS bounding box and the visible matching
    box in the image should be noted in a table which is then used to fix the
    offsets.
  * The `correct_stitching` function of `BboxGenerator` uses the `corr_x` and
    `corr_y` lambda to fix x and y offsets.
  * `BboxGenerator.corr_table` is a reference implementation for an
    interpolating table lookup for every fifth tile.
  * Maybe it could be solved easier with another correction function that can
    be written (dependent on tile X, Y and the original global coordinate,
    resulting in the new global coordinate) and then be supplied to the
    `set_corr_x` and `set_corr_y` functions of the `BboxGenerator` instance.
  * Profit. The alignment of the GDS bounding boxes now match as good as
    possible to the tile images shown with `show_boxes`.
* Now we need to verify the Threshold and cropping of bbox contents so vias
  (white circles) are well visible on standard cells, but not on filler cells.
  This is depending on the algorithm used, we for now focus on algorithm 1.
  * Lower the threshold to 0 or 1. It is measured in the number of white
    pixels from which a cell is displayed for review.
  * Now let it analyze a tile with cells on it, for instance by entering
    `algorithm.analyzeTile(7, 7)`.
  * In case the thresholding-value is wrong, no white pixels could be found or
    the cell has white clouds that should not be there. Usually vias stand out
    very much with a very bright pixel value (200+) compared to the rest.
  * In case the additional cropping is too much or too less, adapt the
    cell crop values with `algorithm.set_cell_crop(...)` to cutout less or more
    of the cropped cell image to not show any adjacent cells in most cases.
  * Finally now also increase the threshold value slowly to reduce the
    false-positive rate of standard cells.
  * Eventually now only "trojan cells" that are labeled as filler cells but
    contain many vias (untypical for filler cells containing essentially
    nothing) trigger a review screen.

Run the Loop:
* Once all parameters are as good as desired (do not fine tune for more than a
  day! ;-) ) let the loop run through all tiles of the chip.
* To do this, either re-run the python script (best still in interactive mode
  just in case something goes wrong), or run `algorithm.analyze()`.
* Every time a suspect cell is found, the cell image as well as the black/white
  thresholded processed image is shown to the user.
  * In case it was probably only a false positive, press any key but one of
    the following in either of the image windows to close them and to continue
    the loop.
  * Press "s" in case the suspect cell should be saved (this saves a by a
    fraction of `algorithm.set_fract(...)` scaled version of the tile with
    every filler cell and the suspect cell in `Config.COLOR_HIGHLIGHT`) and
    shown. After the shown tile is closed with any key the process continues.
  * Press "q" to abort the loop. This might be done to review some parameters
    and to then restart the loop at the current tile, with
    `algorithm.analyze_loop(x, y)` (put the most recent values of the
    "Tile: .." output for x and y)
* The whole analyze Loop should not take too long on a modern laptop (about 30
  minutes at 2000 tiles with a resolution of 4k by 4k with a GDS file of
  400000 bounding boxes).
* Have fun finding the eggs / needle in haystack nearly automatic!