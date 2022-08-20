#!/usr/bin/env python3
import os
import pickle

import numpy as np

class GDSLoader:
    TRANSFORMATION_MATRICES = {
        "R0": np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "MX": np.matrix([[1, 0, 0], [0, -1, 0], [0, 1, 1]]),
        "MY": np.matrix([[-1, 0, 0], [0, 1, 0], [1, 0, 1]]),
        "R180": np.matrix([[-1, 0, 0], [0, -1, 0], [1, 1, 1]]), # same as MXY
        "MXR90": np.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        "R90": np.matrix([[0, 1, 0], [-1, 0, 0], [1, 0, 1]]),
        "R270": np.matrix([[0, -1, 0], [1, 0, 0], [0, 1, 1]]),
        "MYR90": np.matrix([[0, -1, 0], [-1, 0, 0], [1, 1, 1]])
    }


    def __init__(self):
        # initialize empty bboxes
        self.bboxes = []


    def bbox_transform(self, x, y):
        x = (x - self.gds_min_x) / self.gds_width
        y = (y - self.gds_min_y) / self.gds_height
        vector = np.array([x, y, 1])
        res = np.squeeze(np.asarray(np.dot(vector, self.T)))
        return (res[0] / res[2], res[1] / res[2])


    def load_gds(self, filename, image_edges, extra_transformation="R0", use_pickle=True):
        print("loading gds...")

        if use_pickle:
            if os.path.isfile(filename + ".pickle"):
                # if parsed GDS data already exists, load this instead.
                try:
                    with open(filename + ".pickle", "rb") as gds_pickle:
                        self.bboxes, pickle_image_edges, pickle_extra_transformation = pickle.load(gds_pickle)
                    if pickle_image_edges == image_edges and pickle_extra_transformation == extra_transformation:
                        #TODO: also verify integrity (hash?) of the GDSII file as it might also get replaced
                        print("sucessfully unpacked already parsed gds data from pickle file.")
                        return
                    else:
                        print("discard existing pickle file as data which it is based on did change.")
                except ValueError:
                    # pickle seems to be the wrong value, just re-generate
                    print("discard existing pickle file as it seems to have the wrong version.")
                    pass
                except (EOFError, OSError):
                    # pickle seems to be corrupted
                    print("discard existing pickle file as it seems to be corrupted.")
                    pass

        import gdspy
        gdsii = gdspy.GdsLibrary(infile=filename)

        print("loaded gds.")

        top = gdsii.top_level()[0]

        bboxes = []

        # build transformation matrices
        # affine transformation algorithm taken from https://math.stackexchange.com/questions/3037040/normalized-coordinate-of-point-on-4-sided-concave-polygon
        x0 = image_edges[0][0]
        y0 = image_edges[0][1]
        x1 = image_edges[1][0]
        y1 = image_edges[1][1]
        x2 = image_edges[2][0]
        y2 = image_edges[2][1]
        x3 = image_edges[3][0]
        y3 = image_edges[3][1]

        dx1 = x1 - x2
        dx2 = x3 - x2
        dx3 = x0 - x1 + x2 - x3
        dy1 = y1 - y2
        dy2 = y3 - y2
        dy3 = y0 - y1 + y2 - y3
        a13 = (dx3 * dy2 - dy3 * dx2) / (dx1 * dy2 - dy1 * dx2)
        a23 = (dx1 * dy3 - dy1 * dx3) / (dx1 * dy2 - dy1 * dx2)
        a11 = x1 - x0 + a13 * x1
        a12 = y1 - y0 + a13 * y1
        a21 = x3 - x0 + a23 * x3
        a22 = y3 - y0 + a23 * y3

        if extra_transformation not in GDSLoader.TRANSFORMATION_MATRICES:
            raise Exception("Transformation matrix not found. Valid transformations are {}".format(", ".join(GDSLoader.TRANSFORMATION_MATRICES.keys())))
        self.T = GDSLoader.TRANSFORMATION_MATRICES[extra_transformation] @ np.matrix([[a11, a12, a13], [a21, a22, a23], [x0, y0, 1]])

        for element in top:
            if type(element) == gdspy.CellReference:
                bbox = element.get_bounding_box()
                if not bbox is None:
                    bboxes.append((bbox, "FILL" in element.ref_cell.name, str(element)))

        self.gds_min_x = min(min(x[0][0][0], x[0][1][0]) for x in bboxes)
        self.gds_min_y = min(min(x[0][0][1], x[0][1][1]) for x in bboxes)
        self.gds_max_x = max(max(x[0][0][0], x[0][1][0]) for x in bboxes)
        self.gds_max_y = max(max(x[0][0][1], x[0][1][1]) for x in bboxes)

        self.gds_width = self.gds_max_x-self.gds_min_x
        self.gds_height = self.gds_max_y-self.gds_min_y

        self.bboxes = []
        for bbox in bboxes:
            p0 = self.bbox_transform(bbox[0][0][0], bbox[0][0][1])
            p1 = self.bbox_transform(bbox[0][1][0], bbox[0][0][1])
            p2 = self.bbox_transform(bbox[0][1][0], bbox[0][1][1])
            p3 = self.bbox_transform(bbox[0][0][0], bbox[0][1][1])
            poly = (p0, p1, p2, p3)
            polybbox = (min(x[0] for x in poly), min(x[1] for x in poly), max(x[0] for x in poly), max(x[1] for x in poly))
            self.bboxes.append(bbox + ((p0, p1, p2, p3), polybbox))
        
        if use_pickle:
            try:
                with open(filename + ".pickle", "wb") as gds_pickle:
                    pickle.dump((self.bboxes, image_edges, extra_transformation), gds_pickle)
            except Exception as e:
                pass

        print("analyzed bounding boxes and transformed to match stitched images.")