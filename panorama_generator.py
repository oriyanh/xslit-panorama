import cv2 as cv
import numpy as np
from scipy.ndimage import map_coordinates

# Constants
HOMOGRAPHY_DIM = 3


class PanoramaGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, images, images_bw, frame1=0, col1=0, frame2=None, col2=None, angle=None):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.panorama = None
        self.homographies = None
        self.num_frames = len(images)
        self.images = images
        self.images_bw = images_bw
        self.h, self.w = self.images_bw[0].shape
        self.col1 = col1
        self.col2 = col2 if col2 is not None else self.w
        self.frame1 = frame1
        self.frame2 = frame2 if frame2 is not None else self.num_frames
        self.apply_angle(angle, frame1, col1, frame2, col2)
        self.col_order = 1 if self.col2 >= self.col1 else -1
        self.frame_order = 1 if self.frame2 >= self.frame1 else -1
        # self.images = [im[:,::self.col_order,:] for im in images[frame1:frame2]]
        if abs(self.frame1 - self.frame2) <= 1:
            self.images = [images[self.frame1, :, ::self.col_order, :]]
            self.images_bw = [images_bw[self.frame1]]
            self.frame2 = self.frame1
        else:
            # self.images = images[self.frame1:self.frame2:self.frame_order][:, ::self.col_order, :]
            self.images = images[self.frame1:self.frame2:self.frame_order, :, ::self.col_order, :]
            # self.images_bw = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in self.images]
            self.images_bw = images_bw[self.frame1:self.frame2:self.frame_order]
        # self.images = images[frame1:frame2, :, ::self.col_order, :]
        # self.images_bw = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in self.images]
        # self.images_bw = images_bw[frame1:frame2]

    def apply_angle(self, angle, frame1=0, col1=0, frame2=None, col2=None):
        num_frames = self.num_frames
        mid_frame = num_frames // 2
        mid_col = self.w//2
        if angle is None:
            self.angle = None
            self.col1 = col1
            self.col2 = col2  if col2 >=0 else self.w
            self.frame1 = frame1
            self.frame2 = frame2  if frame2 >=0 else self.num_frames
            self.num_frames = abs(self.frame1 - self.frame2)
            # if self.col1 in (0, self.w-1) and self.col2 in (0, self.w-1):
            if self.frame2 == self.frame1:
                angle = 90 if self.col1 > self.col2 else 270
                return
            elif self.col2 == self.col1:
                angle = 0 if self.frame1 < self.frame2 else 180
                return
            else:
                return


        self.angle = angle % 360
        if self.angle == 90:
            self.frame1 = self.frame2 = len(self.images)//2
            self.col1 = self.w
            self.col2 = 0
            return

        if self.angle == 270:
            self.frame1 = self.frame2 = len(self.images)//2
            self.col2 = self.w
            self.col1 = 0
            return

        if 0 <= self.angle <= 45 or 135 <= self.angle <= 225 or 315 <= self.angle < 360:
            if 0 <= self.angle <= 45 or 315 <= self.angle < 360:
                self.frame1 = 0
                self.frame2 = num_frames

            elif 135 <= self.angle <= 225:
                self.frame1 = num_frames
                self.frame2 = 0
            if self.angle == 0 or self.angle == 180:
                self.col1 = self.col2 = mid_col
                return
            if self.angle == 45:
                self.col1 = self.w
                self.col2 = 0
                return
            if self.angle == 225:
                self.col1 = 0
                self.col2 = self.w
                return
            if 135 <= self.angle < 315:
                self.angle = -self.angle
            self.col1 = int(np.tan(self.angle * np.pi / 180.) * mid_col) + mid_col
            self.col2 = -int(np.tan(self.angle * np.pi / 180.) * mid_col) + mid_col
        elif 45 < self.angle < 135:
            self.frame1 =  mid_frame - int(mid_frame / np.tan(angle*np.pi/180.))
            self.frame2 = num_frames - self.frame1
            self.col1 = self.w
            self.col2 = 0
            return
        else:
            if 135 <= self.angle < 315:
                self.angle = -self.angle
            self.frame2 = mid_frame + int(mid_frame / np.tan(angle*np.pi/180.))
            self.angle = abs(self.angle)
            self.frame1 = num_frames - self.frame2
            self.col1 = 0
            self.col2 = self.w
        self.angle = abs(self.angle)

    def find_features(self):

        points_and_descriptors = []
        orb = cv.ORB_create()
        for img in self.images_bw:
            self.h, self.w = img.shape
            kp, desc = orb.detectAndCompute(img, None)
            points_and_descriptors.append((kp, desc))
        return points_and_descriptors


    def calc_homographies(self):
        points_and_descriptors = self.find_features()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            good_points = []
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            matches = bf.knnMatch(desc1, desc2, k=2)
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    good_points.append(m)

            MIN_MATCH_COUNT = 10

            if len(good_points) > MIN_MATCH_COUNT:
                src_pts = np.float32([points1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                dst_pts = np.float32([points2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                # M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=8.0, maxIters=200)
                M, mask = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=8.0, maxIters=200)
                M = np.vstack((M, [0, 0, 1]))
                # M[:2, :2] = np.eye(2)
                Hs.append(M)
            else:
                print("Not enough matches are found - {}/{}".format(len(good_points), MIN_MATCH_COUNT))
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=1)
        self.homographies = self.homographies[self.frames_for_panoramas]


    def generate_panorama(self):

        if abs(self.frame1 - self.frame2) <= 1:
            self.panorama = self.images[0][:, self.col1:self.col2:self.col_order, :]
            return
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=1)
        self.homographies = self.homographies[self.frames_for_panoramas]
        assert self.homographies is not None
        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        # offset = (self.col2 - self.col1) / ((self.frame2 - 1) - self.frame1)
        offset = (self.col2 - self.col1) / abs((self.frame2) - self.frame1)
        if self.col1 == self.col2:
            new_centers = np.array([self.col1]*abs((self.frame2 - 1) - self.frame1))
        else:
            new_centers = np.arange(self.col1, self.col2, step=offset)
        centers = new_centers.round().astype(np.int)[np.newaxis,...]
        centers = np.vstack((centers, [self.h//2]*centers.shape[1])).T
        centers = centers[:,np.newaxis, :]

        warped_centers = [apply_homography(center, h) for center, h in zip(centers, self.homographies)]
        # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
        warped_slice_centers = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]
        panorama_x = int(np.max(warped_slice_centers)) + 1
        panorama_y = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1
        self.panorama = np.zeros((panorama_y[1], panorama_x, 3),
                                 dtype=np.uint8)
        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:-1] + warped_slice_centers[1:]) / 2)
        x_strip_boundary = np.hstack([[max((warped_slice_centers[0] - offset, 0))],
                                      x_strip_boundary,
                                      [panorama_x]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)[np.newaxis,...]
        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1
        width, height = panorama_size
        self.panorama = np.zeros((height, width, 3))
        strips = []
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = self.images[frame_index]
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            # y_bottom = y_offset + warped_image.shape[0]

            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[0, i:i + 2]
            # left_boundary = max((boundaries[0] - x_offset, 0))
            # right_boundary = min((boundaries[1] - x_offset, image.shape[1]))
            # image_strip = warped_image[:, left_boundary:right_boundary][:,::self.col_order,:]
            # strips.append(image_strip)
                # take strip of warped image and paste to current panorama
            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
            image_new = np.vstack([np.zeros((y_offset, image_strip.shape[1], 3), dtype=np.uint8), image_strip[:,::self.col_order],
                       np.zeros((height - image_strip.shape[0] - y_offset, image_strip.shape[1], 3), dtype=np.uint8)])
            # image_new = np.zeros((height, image_strip2.shape[1], 3))
            # image_new[y_offset:y_bottom, :, :] = image_strip2
            # x_end = boundaries[0] + image_strip2.shape[1]
            # self.panorama[y_offset:y_bottom, boundaries[0]:x_end, :] = image_strip2
            strips.append(image_new)
            # x_end = boundaries[0] + image_strip.shape[1]
            # try:
            # self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
            # except ValueError as e:
            #     temp = self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end]
            #     raise e
        # self.panorama = np.hstack(strips)
        self.panorama = np.hstack(strips)

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y]
    point coordinates obtained from transforming pos1 using H12.
    """
    # Homogenizes the (x,y) coordinates to (x,y,1)
    homogenized_pos = np.append(pos1, np.ones((pos1.shape[0], 1)), axis=1)
    pos2 = np.dot(homogenized_pos, H12.T)
    pos_hom = np.zeros(pos1.shape)
    pos_hom[..., 0] = pos2[..., 0]
    pos_hom[..., 1] = pos2[..., 1]
    # Homogenize the new coordinates
    z_axis = pos2[..., 2].reshape((pos2.shape[0], 1))
    pos_hom /= z_axis
    return pos_hom


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_succesive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = np.zeros((len(H_succesive) + 1, HOMOGRAPHY_DIM, HOMOGRAPHY_DIM))
    H2m[m] = np.eye(HOMOGRAPHY_DIM)
    for i in range(m - 1, -1, -1):
        H2m[i] = np.matmul(H2m[i + 1], H_succesive[i])
        H2m[i] /= H2m[i, HOMOGRAPHY_DIM - 1, HOMOGRAPHY_DIM - 1]
    for i in range(m + 1, len(H_succesive)):
        H2m[i] = np.matmul(H2m[i - 1], np.linalg.inv(H_succesive[i - 1]))
        H2m[i] /= H2m[i, HOMOGRAPHY_DIM - 1, HOMOGRAPHY_DIM - 1]
    return [H for H in H2m]


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = apply_homography(np.array([[0, 0], [0, h], [w, 0], [w, h]]), homography)
    corners = np.around(corners).astype(np.int)
    corners = np.array([[np.min(corners[..., 0]), np.min(corners[..., 1])],
                        [np.max(corners[..., 0]), np.max(corners[..., 1])]])
    return corners

def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])

def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # Find corners of bounding box
    top_left, bottom_right = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    # Normalize the inverse homography
    homography_inv = np.linalg.inv(homography)
    homography_inv /= homography_inv[HOMOGRAPHY_DIM - 1, HOMOGRAPHY_DIM - 1]
    # Prepare coordinates for inverse transformation
    x = np.arange(x_min, x_max + 1)
    y = np.arange(y_min, y_max + 1)
    xx, yy = np.meshgrid(x, y)
    transformed_indices = np.dstack((xx, yy))
    points = np.concatenate(transformed_indices)
    # Transform these points, then map the transforemd coords using bilinear interpolation of the original im
    orig_indices = apply_homography(points, homography_inv)
    orig_indices = orig_indices.T
    res = map_coordinates(image, np.array([orig_indices[1], orig_indices[0]]), prefilter=False, order=1)
    res = res.reshape((y_max - y_min + 1, x_max - x_min + 1))
    return res


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if abs(homographies[i][0, -1] - last) > minimum_right_translation and homographies[i][0, -1] != 0:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)
