#!/usr/bin/env python3
from chipsuite.algorithm3 import Algorithm3
from chipsuite.algorithm3_2 import Algorithm3_2
from chipsuite.algorithm3_3 import Algorithm3_3


class Algorithm3_4(Algorithm3_3):
    """
    Algorithm 3.4:
    - use the template matching from algorithm 3.2
    - if considered similar
      - (optimal case: use the matching position for aligning cells over each other)
      - use the via mask based template matching from algorithm 3.3
      - return whether still matching
    - not matching
    """
    def __init__(self, *args, **kwargs):
        self.cell_template_count = {}
        self.use_nth_template = 0
        super().__init__(*args, **kwargs)

    def set_via_correlation_value(self, via_correlation_value):
        self.via_correlation_value = via_correlation_value

    def analyze_bbox(self, croped, identifier):
        pn = Algorithm3.posinv_name(identifier)
        if pn in self.fillers:
            return False, None # too many false positives when filler cells are considered in this algorithm
        if pn not in self.cell_templates:
            # TODO if size is original, store and continue
            #if len(croped) == bh and len(croped[0]) == self.b.bw:
            self.cell_templates[pn] = croped
            self.cell_template_count[pn] = 0
            print(f"template of {pn} added...")
            return False, None

        # correlate with existing template
        print(f"correlate {identifier}...")

        is_diff, output_images = Algorithm3_2.analyze_bbox(self, croped, identifier, identify_cell=False)

        if not is_diff:
            # correlate via masks, we need to sweep again, as there is too much variation
            stored_correlation_value = self.correlation_value
            stored_template = self.cell_templates[pn]
            _, self.cell_templates[pn] = self.generate_via_mask(stored_template)
            self.correlation_value = self.via_correlation_value
            is_diff, via_output_images = super().analyze_bbox(croped, identifier, False)
            self.correlation_value = stored_correlation_value
            self.cell_templates[pn] = stored_template
            if is_diff:
                output_images |= {k: v for k, v in via_output_images.items() if k != "Via Template"}

        if is_diff:
            self.identify_cell(croped, pn, output_images)
        
        if self.cell_template_count[pn] < self.use_nth_template-1:
            self.cell_template_count[pn] += 1
        elif self.cell_template_count[pn] == self.use_nth_template-1:
            self.cell_templates[pn] = croped
            self.cell_template_count[pn] = self.use_nth_template

        return is_diff, output_images

"""for special mode:
    def analyze_bbox(self, croped, identifier):
        pn = Algorithm3.posinv_name(identifier)
        if pn in self.fillers:
            return False, None # too many false positives when filler cells are considered in this algorithm
        if pn not in self.cell_templates:
            # TODO if size is original, store and continue
            #if len(croped) == bh and len(croped[0]) == self.b.bw:
            self.cell_templates[pn] = croped
            self.cell_template_count[pn] = 0
            print(f"template of {pn} added...")
            return False, None

        # correlate with existing template
        print(f"correlate {identifier}...")

        #is_diff, output_images = Algorithm3_2.analyze_bbox(self, croped, identifier, identify_cell=False)

        if True or not is_diff:
            # correlate via masks, we need to sweep again, as there is too much variation
            stored_correlation_value = self.correlation_value
            stored_template = self.cell_templates[pn]
            _, self.cell_templates[pn] = self.generate_via_mask(stored_template)
            self.correlation_value = self.via_correlation_value
            is_diff, via_output_images = super().analyze_bbox(croped, identifier, False)
            self.correlation_value = stored_correlation_value
            self.cell_templates[pn] = stored_template
            #if is_diff:
            #    output_images |= {k: v for k, v in via_output_images.items() if k != "Via Template"}

        #if is_diff:
        #    self.identify_cell(croped, pn, output_images)
        
        if self.cell_template_count[pn] < self.use_nth_template-1:
            self.cell_template_count[pn] += 1
        elif self.cell_template_count[pn] == self.use_nth_template-1:
            self.cell_templates[pn] = croped
            self.cell_template_count[pn] = self.use_nth_template

        return True, None #is_diff, output_images
"""