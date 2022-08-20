#!/usr/bin/env python3
import sys
import time

from chipsuite import GDSLoader, StitchingInfo, BboxGenerator, PowerLineDetector, PowerLineDetector1, Algorithm, Algorithm2, Algorithm3_2, CellIdentifier



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



IMAGE_EDGES = [[1578, 2234], [160553, 1810], [161100, 160601], [2109, 161039]] # 4-edge transformation

gdsloader = GDSLoader()
gdsloader.load_gds("data/65nm/65nm.gds", IMAGE_EDGES, "R90", not BENCHMARK)



stitching_info = StitchingInfo()
stitching_info.load_mist_file("data/65nm/img-global-positions-0.txt")
stitching_info.set_image_folder("data/65nm/tiles")
stitching_info.parse_stitching_info()



bbox_generator = BboxGenerator(gdsloader.bboxes, stitching_info)

# no correction required



powerline_detector = PowerLineDetector1()
powerline_detector.direction = PowerLineDetector.DIR_Y
powerline_detector.averages_height = 210
powerline_detector.set_values(1, 3, 140, 0.8, 0.4)



if ALGORITHM == 2:
    algorithm = Algorithm2(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("65nm_fill_to_std.html")

    algorithm.set_filler_optimal_repeat(27)

    algorithm.set_filler_score_threshold(1000) # effectively disable this, otherwise would be throwing away true positives presumably

    algorithm.set_blur_values(5, 5, True)

    algorithm.set_adaptive_threshold_values(11, -3)

    algorithm.set_erode_dilate_values(3, 3)

    algorithm.set_via_values(4, 10, 60, 0.15, 2)

    algorithm.set_cell_crop(20, 5, 20, 5)
elif ALGORITHM == 3:
    algorithm = Algorithm3_2(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("65nm_std_to_std.html")

    algorithm.set_correlation_value(0.070)

    algorithm.set_cell_identifier(CellIdentifier(algorithm))

    algorithm.prepare()
    algorithm.bboxes_on_edge = False

    algorithm.set_cell_crop(15, -15, 15, -15)

algorithm.set_fract(8) # 1/n for the fused image

algorithm.set_interactive_mode(not BENCHMARK, 0)

algorithm.set_output_folder("output")

algorithm.analyze(Algorithm.MODE_ONLY_CELLS+1 if ALGORITHM >= 3 else Algorithm.MODE_DEFAULT, starttime)



print(f"Completed. Benchmark time of 65nm Algorithm {ALGORITHM}: {time.process_time() - t}")
print(f"Real time of 65nm Algorithm {ALGORITHM}: {time.monotonic() - t2}")

print(f"It is now: {time.strftime('%c')}")