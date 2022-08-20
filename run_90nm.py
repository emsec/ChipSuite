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



IMAGE_EDGES = [[23118, 27288], [141220, 27474], [141051, 145519], [22950, 145322]] # 4-edge transformation

gdsloader = GDSLoader()
gdsloader.load_gds("data/90nm/90nm.gds", IMAGE_EDGES, "R90", not BENCHMARK)



stitching_info = StitchingInfo()
stitching_info.load_mist_file("data/90nm/img-global-positions-0.txt")
stitching_info.set_image_folder("data/90nm/tiles")
stitching_info.parse_stitching_info()



bbox_generator = BboxGenerator(gdsloader.bboxes, stitching_info)

bbox_generator.set_corr_x(lambda x, y, px: round(px - (stitching_info.t_width - x) * y * 170 / (stitching_info.t_width*stitching_info.t_height)))



powerline_detector = PowerLineDetector2()
powerline_detector.set_values(2, 1, 120, 0.9, 0.8)



if ALGORITHM == 2:
    algorithm = Algorithm2(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("90nm_fill_to_std.html")

    algorithm.set_filler_optimal_repeat(27)

    algorithm.set_filler_score_threshold(0.98)

    algorithm.set_blur_values(11, 11, True)

    algorithm.set_adaptive_threshold_values(11, -4)

    algorithm.set_erode_dilate_values(2, 3)

    algorithm.set_via_values(4, 15, 202, 0.6, 1)

    algorithm.set_cell_crop(35, 30, 35, 20)
elif ALGORITHM == 3:
    algorithm = Algorithm3_4(bbox_generator, stitching_info, powerline_detector)
    algorithm.set_html_output("90nm_std_to_std.html")

    algorithm.set_correlation_value(0.123)
    algorithm.set_via_correlation_value(0.006)
    algorithm.set_bsz(10)

    algorithm.set_blur_values(11, 11, True)

    algorithm.set_adaptive_threshold_values(11, -4)

    algorithm.set_erode_dilate_values(2, 3)

    algorithm.set_via_values(4, 15, 202, 0.6, 1)

    algorithm.set_cell_identifier(CellIdentifier(algorithm))

    algorithm.prepare()
    algorithm.bboxes_on_edge = False

    algorithm.set_cell_crop(15, -30, 5, -30)

algorithm.set_fract(8) # 1/n for the fused image

algorithm.set_interactive_mode(not BENCHMARK, 0)

algorithm.set_output_folder("output")

algorithm.analyze(Algorithm.MODE_ONLY_CELLS+1 if ALGORITHM >= 3 else Algorithm.MODE_DEFAULT, starttime, (10, 10))



print(f"Completed. Benchmark time of 90nm Algorithm {ALGORITHM}: {time.process_time() - t}")
print(f"Real time of 90nm Algorithm {ALGORITHM}: {time.monotonic() - t2}")

print(f"It is now: {time.strftime('%c')}")