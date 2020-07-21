import os
import cv2 as cv
import numpy as np

from panorama_generator import PanoramaGenerator

class Engine:

    def __init__(self):
        self.mode = None

        self.src = None
        self.output_dir = None
        self.is_rtl = False
        self.images = []
        self.optical_flows = []
        self._horizontal_motion = 0
        self.output_img = None
        self.motion_min = -20
        self.motion_max = 200
        self.motion = 2.0
        self.reverse_flag = False
        self.percent_of_motions = []
        self.images_bw = None
        self.num_frames = 0


    def load_images(self):  # All images are in BGR order
        assert isinstance(self.src, str)
        assert os.path.exists(self.src)
        del self.images
        self.images = []
        if not os.path.isdir(self.src):  # Is video
            self.output_dir = os.path.join(os.path.basename(self.src), 'output')
            print(f"loading video {self.src}")
            cap = cv.VideoCapture(self.src)
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame
                self.width = frame.shape[1]
                self.images.append(frame)
                frame_num += 1
            cap.release()
        else:  # Is directory holding images
            self.output_dir = os.path.join(self.src, 'output')
            images = sorted(os.listdir(self.src))
            for img_path in images:
                img = cv.imread(os.path.join(self.src, img_path))
                self.width = img.shape[1]
                self.images.append(img)
        self.num_frames = len(self.images)
        print(f"Loaded {self.num_frames} frames")
        self.images_unscaled = np.array(self.images)
        self.images_bw = np.array([cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in self.images_unscaled])
        self.images = self.images_unscaled / 255.0

    def calculate_optical_flow(self):
        # params for lucas-kanade optical flow calculation
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        image_it = iter(self.images)
        old_frame = next(image_it)
        prevgray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(prevgray, mask = None, **feature_params)
        max_flows = []
        for i, img in enumerate(image_it):
            max_flow = 0.
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Lucas-kanade optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
            good_new = p1[st==1]
            # p0 = good_new.reshape(-1,1,2)
            max_flow = max(good_new[..., 0])
            max_flows.append(max_flow)
            prevgray = gray.copy()
            p0 = cv.goodFeaturesToTrack(prevgray, mask = None, **feature_params)

            # Dense optical flow
            # flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # # TODO make sure axis 0 is x-axis in opencv
            # horizontal_flow = flow[..., 0]
            # max_flow = max((max_flow, horizontal_flow.max()))
            # max_flows.append(max_flow)
            # self.optical_flows.append(horizontal_flow)
            # prevgray = gray
        cv.waitKey(1)
        #TODO update min, max motion limits on X-axis, calculate average uniform motion
        self.motion = np.median(max_flows)

    def get_percents(self):
        for i in range(1, self.images_bw.shape[0]):
            motion_between_frames = self.motion_between_frames(self.images_bw[i-1], self.images_bw[i])
            self.percent_of_motions.append(motion_between_frames)
        self.percent_of_motions = np.array(self.percent_of_motions)
        self.percent_of_motions = self.percent_of_motions / np.sum(self.percent_of_motions)

    def find_features(self, img):
        orb = cv.ORB_create()
        points_and_descriptors = orb.detectAndCompute(img, None)
        return points_and_descriptors

    def motion_between_frames(self, image1, image2):
        points_and_descriptors1 = self.find_features(image1)
        points_and_descriptors2 = self.find_features(image2)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        good_points = []
        points1, points2 = points_and_descriptors1[0], points_and_descriptors2[0]
        desc1, desc2 = points_and_descriptors1[1], points_and_descriptors2[1]

        # Find matching feature points.
        matches = bf.knnMatch(desc1, desc2, k=2)
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_points.append(m)

        src_pts = None
        dst_pts = None
        src_pts = np.float32([points1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([points2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        return np.mean(dst_pts-src_pts, axis=1)[0][0]


    def refocus(self, src, is_rtl=False):
        self.mode = 'refocus'
        self.src = src
        self.is_rtl = is_rtl
        self.load_images()
        # if self.is_rtl:
        #     self.images = self.images[::-1]

        # self.calculate_optical_flow()
        self.get_percents()

    def _get_interpolate_image(self, image, fraction):
        frame = np.zeros((image.shape[0], image.shape[1] + 1, 3))

        frame[:, :-1, :] += image*(1-fraction)
        frame[:, 1:, :] += image*fraction

        # new_image = np.


        return frame

    def _interpolate(self, total_motion):
        self.motion = total_motion
        if total_motion > self.motion_max:
            self.motion = self.motion_max
        elif total_motion < self.motion_min:
            self.motion = self.motion_min
        num_images = self.images.shape[0]
        # indexes = np.arange(num_images)
        if ((self.motion < 0) ^ (self.is_rtl)):
            self.motion = np.abs(self.motion)
            # indexes = np.flip(indexes)
            if self.reverse_flag:
                self.images = np.flip(self.images, axis=0)
                self.reverse_flag = True
        elif (not((self.motion < 0) ^ (self.is_rtl))) and self.reverse_flag:
            self.images = np.flip(self.images, axis=0)
            self.reverse_flag = False
        im_shape = self.images[0].shape
        one_motion = self.motion / (num_images-1)

        canvas = np.zeros((im_shape[0], self.images[0].shape[1]+int(np.ceil(self.motion)) + 1, 3))
        count_arr = np.zeros(self.images[0].shape[1]+int(np.ceil(self.motion)) + 1)

        canvas[:, : im_shape[1], :] = self.images[0]
        count_arr[:im_shape[1]] += 1
        counter = 0
        for i in range(1, self.images.shape[0]):
            curr_motion = np.sum(self.percent_of_motions[:i])*self.motion
            # counter += self.percent_of_motions*self.motion
            fraction = curr_motion - int(curr_motion)
            inter = self._get_interpolate_image(self.images[i], fraction)
            canvas[:, int(curr_motion): int(curr_motion) + im_shape[1] + 1, :] += inter
            count_arr[int(curr_motion): int(curr_motion) + im_shape[1] + 1] += 1

        self.output_img = canvas / count_arr[np.newaxis, :,np.newaxis]


    @property
    def motion(self):
        return self._horizontal_motion

    @motion.setter
    def motion(self, motion):
        self._horizontal_motion = motion

    def init_viewpoint(self):
        self.height, self.width, _ = self.images[0].shape
        self.canonical_gen = PanoramaGenerator(self.images_unscaled, self.images_bw, 0, 0, len(self.images_unscaled), self.width)
        self.canonical_gen.calc_homographies()


    def change_viewpoint(self, src, frame1=None, col1=None, frame2=None, col2=None, angle=None, is_rtl=False):
        self.mode = 'viewpoint'
        self.src = src
        self.is_rtl = is_rtl
        if self.is_rtl:
            self.images = self.images[::-1]
        self.col1 = col1 if col1 is not None else 0
        self.col2 = col2 if col2 is not None else self.width
        self.frame1 = frame1 if frame1 is not None else 0
        self.frame2 = frame2 if frame2 is not None else len(self.images)
        self.num_frames = max((abs(self.frame1 - self.frame2), 1))
        self.angle = angle
        self._panorama()


    def _panorama(self):
        self.pano_gen = PanoramaGenerator(self.images_unscaled, self.images_bw, self.frame1, self.col1, self.frame2, self.col2, self.angle)
        self.frame1 = self.pano_gen.frame1
        self.frame2 = self.pano_gen.frame2
        self.col1 = self.pano_gen.col1
        self.col2 = self.pano_gen.col2
        self.angle = self.pano_gen.angle
        self.pano_gen.homographies = self.canonical_gen.homographies[self.pano_gen.frame1:self.pano_gen.frame2:self.pano_gen.frame_order]
        self.pano_gen.generate_panorama()
        self.output_img = self.pano_gen.panorama
