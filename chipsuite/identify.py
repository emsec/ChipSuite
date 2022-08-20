#!/usr/bin/env python3
from chipsuite.algorithm3_2 import Algorithm3_2


class CellIdentifier:
    """
    Will only work for algorithms inheriting from 3.2 (3.2, 3.3, 3.4)

    - iterate through all templates
    - ignore these with differing orientation (such that VCC / GND rails would get swapped)
    - ignore these that differ too much in their shape by a given factor
    - the one with the best score of the ones left wins
    """
    def __init__(self, algorithm):
        assert isinstance(algorithm, Algorithm3_2), "cell templates and template matching functions from Algorithm 3.2 required"
        self.a = algorithm
        self.shape_factor = 0.05


    def set_shape_factor(self, shape_factor):
        self.shape_factor = shape_factor


    @staticmethod
    def is_cell_rotated(pn):
        return ("rotation None" in pn) ^ ("reflection False" in pn)


    def identify_cell(self, image, pn):
        orientation = self.is_cell_rotated(pn)
        #im1 = Algorithm3_2.image_template_pad(Algorithm3_2.image_preprocess(image))

        min_corr = 1.0
        best_match = None

        for cell in self.a.cell_templates.keys():
            # test what happens now? maybe false positives get detected as such
            """if cell == pn:
                # skip the obviously wrong candidate
                continue"""
            
            if self.is_cell_rotated(cell) != orientation:
                # skip differing orientation
                continue

            compare = self.a.cell_templates[cell]

            if (abs(compare.shape[1] - image.shape[1])) / (image.shape[1]) > self.shape_factor or \
                (abs(compare.shape[0] - image.shape[0])) / (image.shape[0]) > self.shape_factor:
                # skip differing size
                continue

            #im2 = Algorithm3_2.image_preprocess(compare)

            #corr, _, _ = self.a.template(im1, im2, quiet=True)
            min_cell_corr = 1.0
            for corr, _ in self.a.sweep_correlate(compare, image, quiet=True):
                if corr < min_cell_corr:
                    min_cell_corr = corr
            #print(f"Identify: Match cell against {cell}: {corr}")

            if min_corr > min_cell_corr:
                min_corr = min_cell_corr
                best_match = cell
        
        return best_match, min_corr