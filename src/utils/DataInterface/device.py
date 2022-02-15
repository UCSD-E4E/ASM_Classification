import os
import csv
import abc
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Device(ABC):
    def __init__(self, uuid, desc, fw_ver, loc, loc_units, type, datapath):
        self.uuid = uuid
        self.description = desc
        self.firmware_ver = fw_ver
        self.location = loc
        self.location_units = loc_units
        self.type = type
        self.datapath = datapath
        self.labelpath = datapath

    def data(self):
        files = []
        for file in os.listdir(self.datapath):
            if file.endswith(".mp4"):
                files.append(self.uuid + "/" +  str(file))
        return files

    def dataLabels(self):
        label_csv = open(self.labelpath)
        reader = csv.reader(label_csv, delimiter=",")
        labels = []
        for row in reader:
            label = (datetime.fromisoformat(row[0]), row[1])
            labels.append(label)
        return labels

    def metadata(self):
        pass

    def dataPeriod(self):
        pass


class OnBoxDevice(Device):
    def __init__(self, uuid, desc, fw_ver, loc, loc_units, type, datapath, labelpath):
        super().___init___(uuid, desc, fw_ver, loc, loc_units, type, datapath)
        self.labelpath = labelpath


class RemoteSensorDevice(Device):
    pass