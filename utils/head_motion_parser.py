import argparse
import csv
import sys

from utils.vector_util import Vector3


class HeadMotionParser:

    def __init__(self):
        pass

    def read_framelog(self, log_path):

        head_mvmts = []

        with open(log_path, "r") as log_file:
            lines = csv.reader(log_file, delimiter=",")
            for line in lines:

                if "VD-FRAME" in line[0] and ":" in line[0]:
                    _, time_stamp, x_val = line[0].split(": ")
                    line[0] = x_val
                elif ":" in line[0]:
                    continue

                vect3 = Vector3()
                vect3.x = float(line[0])
                vect3.y = float(line[1])
                vect3.z = float(line[2])

                head_mvmts.append(vect3)

        return head_mvmts


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Update the values -> --head_data = head movement file")
        exit(0)

    # parse all arguments
    # create arg parser
    arg_parser = argparse.ArgumentParser()

    # add argument elements to argparse
    arg_parser.add_argument("-hd", "--head_data", required=True, help="add txt/csv file containing head movements")

    _args = vars(arg_parser.parse_args())

    mvmt_file_path = _args["head_data"]
    parser = HeadMotionParser()
    mvmt_data = parser.read_framelog(mvmt_file_path)
