import csv


class HeadMotionParser:

    def read_framelog(self, log_path):
        with open(log_path, 'r') as log_file:
            lines = csv.reader(log_file, delimiter=",")
