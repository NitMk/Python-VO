import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging

from utils.PinholeCamera import PinholeCamera



class SequenceImageLoaderTunnel(object):
    default_config = {
        "root_path": " /home/nitzan/Thesis/ThesisRepo/bag14/cam2/ImgsValidClean/",
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
        self.img_N = self.img_N + 2000 #just because the ordering is messed up
        #for cam 1 (lower)
        # self.cam = PinholeCamera(1280.0, 720.0, 1014.8030544556, 1013.7896293084, 627.2438643441, 429.880932612)
        #for cam 2 (upper)
        self.cam = PinholeCamera(1280.0, 720.0, 1579.6093592162, 1567.4325689515, 643.7388676583, 294.5986817384)


    def __getitem__(self, item):
        numstr = "frame%06i" % item
        file_name = self.config["root_path"]  + numstr + "." + self.config["format"]
        # file_name = self.config["root_path"] + "/" + str(item) + "." + self.config["format"]
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

    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]
