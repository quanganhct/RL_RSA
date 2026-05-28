import csv
import typing
import os

class CSVWriter:
    def __init__(self, filename, foldername) -> None:
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        self.file = open("%s/%s"%(foldername, filename), 'w', newline='')
        self.writer = csv.writer(self.file)

    def close(self):
        self.file.close()

    def write(self, row:typing.List):
        self.writer.writerow(row)
        self.file.flush()