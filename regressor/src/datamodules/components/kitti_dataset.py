"""
Dataset modules for load kitti dataset and convert to yolo3d format
"""

from pathlib import Path

import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.utils import Calib as calib
from src.utils.ClassAverages import ClassAverages

class KITTIDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = '../data/KITTI/training',
        dataset_sets: str = '../data/KITTI/training/train.txt', # [train.txt, val.txt]
        bins: int = 2,
        overlap: float = 0.1,
    ):
        super().__init__()

        # dataset path
        dataset_path = Path(dataset_path)
        self.image_path = dataset_path / "images"  # image_2
        self.label_path = dataset_path / "label_2"
        self.calib_path = dataset_path / "calib"
        self.global_calib = dataset_path / "calib_kitti.txt"
        self.dataset_sets = Path(dataset_sets)

        # set projection matrix
        self.proj_matrix = calib.get_P(self.global_calib)

        # index from images_path
        self.sets = open(self.dataset_sets, 'r')
        self.ids = [id.split('\n')[0] for id in self.sets.readlines()]
        # self.ids = [x.split(".")[0] for x in sorted(os.listdir(self.image_path))]
        
        self.num_images = len(self.ids)

        # set ANGLE BINS
        self.bins = bins
        self.angle_bins = self.generate_bins(self.bins)
        self.interval = 2 * np.pi / self.bins
        self.overlap = overlap

        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(
                (
                    (i * self.interval - overlap) % (2 * np.pi),
                    (i * self.interval + self.interval + overlap) % (2 * np.pi),
                )
            )

        # AVERANGE num classes dataset
        # class_list same as in detector
        self.class_list = ["Car", "Pedestrian", "Cyclist", "Truck"]
        self.averages = ClassAverages(self.class_list)

        # list of object [id (000001), line_num]
        self.object_list = self.get_objects(self.ids)

        # label: contain image label params {bbox, dimension, etc}
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id
            self.labels[id][str(line_num)] = label

        # current id and image
        self.curr_id = ""
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            # read image (.png)
            self.curr_img = cv2.imread(str(self.image_path / f"{id}.png"))

        label = self.labels[id][str(line_num)]

        obj = DetectedObject(
            self.curr_img, label["Class"], label["Box_2D"], self.proj_matrix, label=label
        )

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    # def generate_sets(self, sets_file):
    #     with open(self.dataset_sets) as file:
    #         for line_num, line in enumerate(file):
    #             ids = line

    def generate_bins(self, bins):
        angle_bins = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            angle_bins[i] = i * interval
        angle_bins += interval / 2  # center of bins

        return angle_bins

    def get_objects(self, ids):
        """Get objects parameter from labels, like dimension and class name."""
        objects = []
        for id in ids:
            with open(self.label_path / f"{id}.txt") as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(" ")
                    obj_class = line[0]
                    if obj_class not in self.class_list:
                        continue

                    dimension = np.array(
                        [float(line[8]), float(line[9]), float(line[10])], dtype=np.double
                    )
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))

        self.averages.dump_to_file()
        return objects

    def get_label(self, id, line_num):
        lines = open(self.label_path / f"{id}.txt").read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2 * np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(" ")

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        # Alpha is orientation will be regressing
        # Alpha = [-pi, pi]
        Alpha = line[3]
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        # Dimension: height, width, length
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)
        # modify the average
        Dimension -= self.averages.get_item(Class)

        # Locattion: x, y, z
        Location = [line[11], line[12], line[13]]
        # bring the KITTI center up to the middle of the object
        Location[1] -= Dimension[0] / 2

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # angle on range [0, 2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
            "Class": Class,
            "Box_2D": Box_2D,
            "Dimensions": Dimension,
            "Alpha": Alpha,
            "Orientation": Orientation,
            "Confidence": Confidence,
        }

        return label


class DetectedObject:
    """Processing image for NN input."""

    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        # check if proj_matrix is path
        if isinstance(proj_matrix, str):
            proj_matrix = calib.get_P(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """Calculate global angle of object, see paper."""
        width = img.shape[1]
        # Angle of View: fovx (rad) => 3.14
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):
        # transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        process = transforms.Compose([transforms.ToTensor(), normalize])

        # crop image
        pt1, pt2 = box_2d[0], box_2d[1]
        crop = img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

        # apply transform for batch
        batch = process(crop)

        return batch