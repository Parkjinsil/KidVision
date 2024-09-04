from tools.canvas_detection import canvas_detect
from tools.handling_yolos import handling_yolos, get_accuracy
import os

input_dir = r"{}".format(input("input_dir: "))
output_dir = r"{}".format(input("output_dir: "))
csv_folder_path = r"{}".format(input("csv_folder_path: "))

# input_dir = r"data/test"
# output_dir = r"data/test_dst"
# csv_folder_path = r"data/test_dstdst"

classes = ["house", "tree", "person"]
canvas_paths = {}

for cat in classes:
    join_path = os.path.join(output_dir, cat)
    if os.path.exists(join_path) == False:
        os.mkdir(join_path)
    canvas_paths[cat] = join_path

canvas_detect(input_dir, output_dir)

handling_yolos(classes, canvas_paths, csv_folder_path)

# get_accuracy()