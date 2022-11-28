import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging

from utils.PinholeCamera import PinholeCamera



class SequenceImageLoader(object):
    default_config = {
        "root_path": " /home/nitzan/Thesis/ThesisRepo/bag14/phone_vid/",
        "start": 0,
        "format": "png"
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Sequence image loader config: ")
        logging.info(self.config)

        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/*." + self.config["format"]))
        self.cam = PinholeCamera(640.0, 4801.0, 525, 525, 319.5, 239.5)


    def __getitem__(self, item):
        file_name = self.config["root_path"] + "/" + str(item) + "." + self.config["format"]
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            img = self.__getitem__(self.img_id)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
