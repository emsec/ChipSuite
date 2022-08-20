#!/usr/bin/env python3
import sys
import time

from chipsuite import GDSLoader, StitchingInfo, BboxGenerator, PowerLineDetector2, Algorithm, Algorithm2, Algorithm3_4, CellIdentifier



BENCHMARK = True
if sys.argv[1] == "fill":
    ALGORITHM = 2
elif sys.argv[1] == "std":
    ALGORITHM = 3
else:
    print('Unsupported algorithm argument. Possible options are: "fill", "std".', file=sys.stderr)
    sys.exit(1)



t = time.process_time()
t2 = time.monotonic()
print(f"It is now: {time.strftime('%c')}")
starttime = time.strftime('%c')



IMAGE_EDGES = [[542, 3419], [209110, 2996], [209910, 210897], [1443, 211108]] # 4-edge transformation

gdsloader = GDSLoader()
gdsloader.load_gds("data/40nm/40nm.gds", IMAGE_EDGES, "R180", not BENCHMARK)



stitching_info = StitchingInfo()
stitching_info.load_mist_file("data/40nm/img-global-positions-0.txt")
stitching_info.set_image_folder("data/40nm/tiles")
stitching_info.parse_stitching_info()



bbox_generator = BboxGenerator(gdsloader.bboxes, stitching_info)

CORR_X_DATA = [
    [0, -41, -51, -29, -41, 7, 26, 18, 32, 13, 23, 4, 0],
    [-65, -52, -49, -35, -26, -11, 5, 21, 4, 1, -11, -31, -42],
    [-79, -67, -55, -44, -41, -36, -35, -11, -16, -17, -27, -40, -45],
    [-100, -72, -61, -43, -36, -36, -23, -14, -14, -25, -26, -32, -35],
    [-94, -71, -60, -46, -30, -29, -18, -15, -9, -16, -25, -40, -50],
    [-99, -70, -50, -38, -29, -20, -8, -11, -9, -15, -14, -32, -24],
    [-74, -58, -48, -30, -20, -9, -7, -3, -3, -13, -24, -31, -43],
    [-76, -55, -36, -23, -9, -3, 2, 3, 3, -1, -13, -28, -37],
    [-69, -50, -28, -12, 0, 7, 9, 28, 15, 11, -1, -14, -24],
    [-73, -38, -22, -8, 20, 20, 22, 25, 9, 23, 12, 5, 13],
    [-51, -31, -12, 4, 19, 26, 33, 41, 39, 31, 22, 12, 7],
    [-35, -15, 3, 21, 27, 31, 47, 56, 45, 34, 21, 3, -6],
    [0, -1, 12, 33, 32, 35, 54, 52, 50, 31, 12, 0, 6],
]

CORR_Y_DATA = [
    [0, -13, -11, -4, 3, 13, 19, 21, 31, 23, 21, 4, 0],
    [0, 10, 11, 17, 21, 30, 35, 39, 40, 40, 33, 24, 20],
    [19, 11, 19, 25, 31, 35, 42, 41, 44, 44, 38, 26, -1],
    [-12, 11, 21, 28, 35, 39, 39, 47, 47, 51, 49, 29, -24],
    [6, 17, 22, 27, 33, 38, 46, 47, 52, 57, 61, 81, 135],
    [18, 20, 25, 30, 35, 41, 46, 50, 56, 62, 69, 79, 84],
    [13, 21, 25, 31, 33, 39, 43, 49, 55, 60, 70, 86, 114],
    [24, 21, 25, 27, 29, 34, 38, 42, 50, 55, 63, 72, 75],
    [7, 21, 21, 26, 29, 30, 31, 37, 40, 42, 48, 53, 50],
    [20, 16, 20, 20, 21, 22, 24, 28, 26, 29, 33, 34, 42],
    [8, 14, 15, 13, 12, 12, 9, 8, 12, 14, 15, 8, -28],
    [26, 7, 7, 0, 2, 0, -11, -14, -9, -13, -10, 8, 26],
    [0, -22, -6, -34, -18, -17, -50, -35, -30, -30, -48, -15, -8],
]

CORR_X_EXTRA = {
    (21, 0): 17,
    (31, 9): 17,
    (31, 10): 12
}

bbox_generator.set_corr_x(lambda x, y, px: BboxGenerator.corr_table(x, y, px, CORR_X_DATA) if (x, y) not in CORR_X_EXTRA else px - CORR_X_EXTRA[x, y])
bbox_generator.set_corr_y(lambda x, y, py: BboxGenerator.corr_table(x, y, py, CORR_Y_DATA))



powerline_detector = PowerLineDetector2()
powerline_detector.set_values(4, 1, 130, 0.9, 0.71)



if ALGORITHM == 2:
    algorithm = Algorithm2(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("40nm_fill_to_std.html")

    algorithm.set_filler_optimal_repeat(27) # this value is the usual width between two horizontal repititions in the filler cells

    algorithm.set_filler_score_threshold(0.98)
elif ALGORITHM == 3:
    algorithm = Algorithm3_4(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("40nm_std_to_std.html")

    algorithm.set_correlation_value(0.177)
    algorithm.set_via_correlation_value(0.031)

    algorithm.set_cell_identifier(CellIdentifier(algorithm))

    algorithm.prepare()
    algorithm.bboxes_on_edge = False

    algorithm.set_cell_crop(-50, 0, -50, 20)

algorithm.set_blur_values(23, 11, True) # TODO: why is X still doubled?

algorithm.set_adaptive_threshold_values(11, -4)

algorithm.set_erode_dilate_values(2, 3)

algorithm.set_via_values(2, 12, 202, 0.6, 2)

algorithm.set_fract(8) # 1/n for the fused image

algorithm.set_interactive_mode(not BENCHMARK, 0)

algorithm.set_output_folder("output")

algorithm.analyze(Algorithm.MODE_ONLY_CELLS+1 if ALGORITHM >= 3 else Algorithm.MODE_DEFAULT, starttime)



print(f"Completed. Benchmark time of 40nm Algorithm {ALGORITHM}: {time.process_time() - t}")
print(f"Real time of 40nm Algorithm {ALGORITHM}: {time.monotonic() - t2}")

print(f"It is now: {time.strftime('%c')}")