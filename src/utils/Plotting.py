import cv2
import numpy as np
from enum import Enum
import itertools

from .Calib import *
from .Math import *
from src.utils import Calib


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)


def constraint_to_color(constraint_idx):
    return {
        0: cv_colors.PURPLE.value,  # left
        1: cv_colors.ORANGE.value,  # top
        2: cv_colors.MINT.value,  # right
        3: cv_colors.YELLOW.value,  # bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2] / point[2]
    point = point.astype(np.int16)

    return point


# take in 3d points and plot them on image as red circles
def plot_3d_pts(
    img,
    pts,
    center,
    calib_file=None,
    cam_to_img=None,
    relative=False,
    constraint_idx=None,
):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j, i in enumerate(pt)]  # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)


def plot_3d_box(img, cam_to_img, ry, dimension, center):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    # LINE
    cv2.line(
        img,
        (box_3d[0][0], box_3d[0][1]),
        (box_3d[2][0], box_3d[2][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[4][0], box_3d[4][1]),
        (box_3d[6][0], box_3d[6][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[0][0], box_3d[0][1]),
        (box_3d[4][0], box_3d[4][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[2][0], box_3d[2][1]),
        (box_3d[6][0], box_3d[6][1]),
        cv_colors.GREEN.value,
        2,
    )

    cv2.line(
        img,
        (box_3d[1][0], box_3d[1][1]),
        (box_3d[3][0], box_3d[3][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[1][0], box_3d[1][1]),
        (box_3d[5][0], box_3d[5][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[7][0], box_3d[7][1]),
        (box_3d[3][0], box_3d[3][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[7][0], box_3d[7][1]),
        (box_3d[5][0], box_3d[5][1]),
        cv_colors.GREEN.value,
        2,
    )

    for i in range(0, 7, 2):
        cv2.line(
            img,
            (box_3d[i][0], box_3d[i][1]),
            (box_3d[i + 1][0], box_3d[i + 1][1]),
            cv_colors.GREEN.value,
            2,
        )

    # frame to drawing polygon
    frame = np.zeros_like(img, np.uint8)

    # front side
    cv2.fillPoly(
        frame,
        np.array(
            [[[box_3d[0]], [box_3d[1]], [box_3d[3]], [box_3d[2]]]], dtype=np.int32
        ),
        cv_colors.BLUE.value,
    )

    alpha = 0.5
    mask = frame.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]


def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

def calc_theta_ray(img_width, box_2d, proj_matrix):
    """Calculate global angle of object, see paper."""

    # check if proj_matrix is path
    if isinstance(proj_matrix, str):
        proj_matrix = Calib.get_P(proj_matrix)

    # Angle of View: fovx (rad) => 3.14
    fovx = 2 * np.arctan(img_width / (2 * proj_matrix[0][0]))
    # center_x = (box_2d[1][0] + box_2d[0][0]) / 2
    center_x = ((box_2d[2] - box_2d[0]) / 2 ) + box_2d[0]
    dx = center_x - (img_width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan((2 * dx * np.tan(fovx / 2)) / img_width)
    angle = angle * mult

    return angle

def calc_alpha(orient, conf, bins=2):
    angle_bins = generate_bins(bins=bins)
    
    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += angle_bins[argmax]
    alpha -= np.pi

    return alpha

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of bins

    return angle_bins