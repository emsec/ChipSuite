#!/usr/bin/env python3
import os

import cv2


class StitchingInfo:
    def __init__(self):
        pass


    @staticmethod
    def __conv(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v.replace(",", "."))
            except ValueError:
                return v


    def load_mist_file(self, stitching_file):
        """
        Info: The MIST file is of following structure (each tile one line of these):

        file: Tile_007-007-000000_0-000.s0001_e00.tif; corr: 0,7687172905; position: (22405, 23555); grid: (6, 6);
        
        """
        self.stitching = {}
        with open(stitching_file, "r") as f:
            for line in f:
                tile = {}
                for part in line.strip("\r\n;").split("; "):
                    key, value = part.split(": ", 1)
                    if value[0] == "(" and value[-1] == ")":
                        value = tuple(StitchingInfo.__conv(x) for x in value[1:-1].split(", "))
                    else:
                        value = StitchingInfo.__conv(value)
                    tile[key] = value
                self.stitching[tile["grid"]] = tile


    def set_image_folder(self, image_folder=""):
        self.image_folder = image_folder


    def parse_stitching_info(self):
        # minimum is 0,0 so we don't need min/max values here
        self.st_width = max(x["position"][0] for x in self.stitching.values())
        self.st_height = max(x["position"][1] for x in self.stitching.values())

        # tiles
        if "demo" in self.image_folder:
            self.t_width = list(self.stitching.keys())[0][0] + 1
            self.t_height = list(self.stitching.keys())[0][1] + 1
            self.t_count = 1
        else:
            self.t_width = max(x[0] for x in self.stitching.keys())+1
            self.t_height = max(x[1] for x in self.stitching.keys())+1
            self.t_count = self.t_width * self.t_height
            assert self.t_count == len(self.stitching), "Stitching textfile seems to be incomplete, as not every position in the grid has exactly one entry in the file"

        # take first tile's size as a global tile size (all tiles are of the same size!)
        image = cv2.imread(os.path.join(self.image_folder, list(self.stitching.values())[0]["file"]))
        self.pw, self.ph = (len(image[0]), len(image))

        print("parsed stitching information.")