import argparse
import csv
import sys


class HeadMotionParser:

    def __init__(self):
        pass

    def read_framelog(self, log_path):
        with open(log_path, "r") as log_file:
            lines = csv.reader(log_file, delimiter=",")
            for line in lines:
                print(line)


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
    parser.read_framelog(mvmt_file_path)
