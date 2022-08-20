#!/usr/bin/env python3
import sys
import time

from chipsuite import GDSLoader, StitchingInfo, BboxGenerator, FakePowerLineDetector, PowerLineDetector, Algorithm, Algorithm2, Algorithm3_4, CellIdentifier



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



IMAGE_EDGES = [[2965, 3905.98], [204515.97, 3232], [205260.64, 204506.74], [3518.89, 205231.10]] # 4-edge transformation

gdsloader = GDSLoader()
gdsloader.load_gds("data/28nm/28nm.gds", IMAGE_EDGES, "R180", not BENCHMARK)



stitching_info = StitchingInfo()
stitching_info.load_mist_file("data/28nm/img-global-positions-0.txt")
stitching_info.set_image_folder("data/28nm/tiles")
stitching_info.parse_stitching_info()



bbox_generator = BboxGenerator(gdsloader.bboxes, stitching_info)

CORR_X_DATA = [
    [0, -30, -48, -53, -66, -83, -80, -60, -43, -27, 6, 0],
    [-10, -24, -52, -48, -66, -60, -62, -54, -32, -4, 26, 37],
    [1, -29, -37, -52, -51, -79, -51, -41, -21, -3, 21, 49],
    [1, -13, -45, -52, -58, -63, -61, -43, -35, -16, 12, 51],
    [16, 3, -50, -57, -61, -71, -51, -35, -17, -9, 17, 44],
    [7, -13, -45, -39, -58, -64, -56, -36, -35, -15, 1, 20],
    [-7, -21, -30, -42, -50, -60, -70, -51, -30, -8, 7, 27],
    [-5, -4, -41, -46, -57, -56, -76, -66, -42, -35, 6, 33],
    [0, -12, -53, -46, -66, -77, -85, -82, -58, -35, -35, 7],
    [-13, -22, -33, -49, -60, -75, -73, -59, -59, -52, -24, -12],
    [-6, -16, -23, -33, -49, -59, -44, -50, -49, -47, -27, -13],
    [0, 3, -22, -40, -51, -52, -56, -45, -37, -41, -26, -4],
]

CORR_Y_DATA = [
    [0, 0, -10, -20, -21, -21, -21, -18, -16, -4, 0, 0],
    [-41, -45, -48, -51, -52, -52, -50, -45, -42, -37, -34, -34],
    [-68, -72, -69, -68, -70, -69, -67, -63, -63, -60, -59, -56],
    [-85, -84, -86, -85, -82, -81, -81, -77, -75, -73, -75, -70],
    [-96, -92, -90, -88, -89, -88, -85, -83, -80, -79, -79, -76],
    [-100, -95, -94, -90, -93, -89, -89, -85, -85, -84, -83, -80],
    [-97, -97, -94, -93, -93, -91, -92, -86, -87, -86, -84, -82],
    [-89, -90, -86, -86, -83, -84, -81, -79, -82, -81, -79, -79],
    [-80, -78, -76, -75, -74, -72, -69, -69, -69, -67, -70, -67],
    [-64, -62, -59, -56, -53, -53, -52, -53, -54, -52, -53, -53],
    [-34, -36, -32, -32, -30, -29, -27, -29, -30, -29, -28, -30],
    [0, 2, 1, 5, 7, 6, 6, 6, 4, 5, 1, 0],
]

bbox_generator.set_corr_x(lambda x, y, px: BboxGenerator.corr_table(x, y, px, CORR_X_DATA) + (20 if x == 51 and y == 38 else 0))
bbox_generator.set_corr_y(lambda x, y, py: BboxGenerator.corr_table(x, y, py, CORR_Y_DATA))



# no power line detector, as it yields to too many wrong detections 
powerline_detector = FakePowerLineDetector(PowerLineDetector.DIR_X)



if ALGORITHM == 2:
    algorithm = Algorithm2(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("28nm_fill_to_std.html")

    algorithm.set_filler_optimal_repeat(27) # this value is the usual width between two horizontal repititions in the filler cells

    algorithm.set_filler_score_threshold(0.38)

    # X-Blur is doubled as there are these annoying vertical "fences" in the 28nm chip
    algorithm.set_blur_values(23, 11, True)

    algorithm.set_adaptive_threshold_values(11, -4)

    algorithm.set_erode_dilate_values(3, 3)

    algorithm.set_via_values(2, 15, 202, 0.6, 1)

    algorithm.set_cell_crop(10, 20, 30, 35)
elif ALGORITHM == 3:
    algorithm = Algorithm3_4(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("28nm_std_to_std.html")

    algorithm.set_correlation_value(1.0)
    algorithm.set_via_correlation_value(0.047)

    algorithm.set_blur_values(23, 11, True)

    algorithm.set_adaptive_threshold_values(7, -1)

    algorithm.set_erode_dilate_values(2, 3)

    algorithm.set_via_values(2, 15, 50, 0.6, 1)

    algorithm.set_cell_identifier(CellIdentifier(algorithm))

    algorithm.prepare()
    algorithm.bboxes_on_edge = False

    algorithm.set_cell_crop(-15, 10, -15, 10)

algorithm.set_fract(8) # 1/n for the fused image

algorithm.set_interactive_mode(not BENCHMARK, 0)

algorithm.set_output_folder("output")

algorithm.analyze(Algorithm.MODE_ONLY_CELLS+1 if ALGORITHM >= 3 else Algorithm.MODE_DEFAULT, starttime, (10, 10))



print(f"Completed. Benchmark time of 28nm Algorithm {ALGORITHM}: {time.process_time() - t}")
print(f"Real time of 28nm Algorithm {ALGORITHM}: {time.monotonic() - t2}")

print(f"It is now: {time.strftime('%c')}")