import cv2
import numpy as np
import myopencv as my

class Line:
    def __init__(self):
        self.start_x = None
        self.n_window = 10
        self.window_margin = 80
        self.min_num_pixel = 30
        self.radius_of_curvature = None
        self.coord = list()

        """
        self.detected = False
        self.prevx = []
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None
        """

def init_start_x(bin_img, left_lane, right_lane):
    h_img, w_img = bin_img.shape[:2]

    hist = np.sum(bin_img[int(h_img/2):, :], axis=0)
    mid_x = np.int(w_img/2)

    left_lane.start_x = np.argmax(hist[:mid_x])
    right_lane.start_x = np.argmax(hist[mid_x:]) + mid_x


def serch_lane_point(bin_img, left_lane, right_lane, plot=False):
    h_img, w_img = bin_img.shape[:2]
    center_of_lane = []

    n_window = left_lane.n_window
    window_margin = left_lane.window_margin
    min_num_pixel = left_lane.min_num_pixel
    h_window = int(h_img/n_window)

    winL_center = left_lane.start_x                            #; print('L_start', winL_center)
    winR_center = right_lane.start_x                           #; print('R_start', winR_center)
    left_lane.coord.append((winL_center, h_img - 1))           #; print('window', n, 'LC', winL_center)
    right_lane.coord.append((winR_center, h_img - 1))          #; print('window', n, 'RC', winR_center)

    for n in range(n_window):
        win_bound_U = h_img - (n + 1) * h_window               #; print('window', n, 'U', win_bound_U)
        win_bound_D = h_img - n * h_window                     #; print('window', n, 'D', win_bound_U)
        winL_bound_L = winL_center - window_margin             #; print('window', n, 'LL', winL_bound_L)
        winL_bound_R = winL_center + window_margin             #; print('window', n, 'LR', winL_bound_R)
        winR_bound_L = winR_center - window_margin             #; print('window', n, 'RL', winR_bound_L)
        winR_bound_R = winR_center + window_margin             #; print('window', n, 'RR', winR_bound_R)

        if winL_bound_L < 0:
            winL_bound_L = 0                                   #; print('window', n, 'new LL', winL_bound_L)
        if winL_bound_R > w_img - 1:
            winL_bound_R = w_img - 1                           #; print('window', n, 'new LR', winL_bound_R)
        if winR_bound_L < 0:
            winR_bound_L = 0                                   #; print('window', n, 'new RL', winR_bound_L)
        if winR_bound_R > w_img - 1:
            winR_bound_R = w_img - 1                           #; print('window', n, 'new RR', winR_bound_R)

        winL = bin_img[win_bound_U:win_bound_D, winL_bound_L:winL_bound_R]
        if np.count_nonzero(winL) > min_num_pixel:
            winL_hist = np.sum(winL, axis=0)                                          #; print('-----L hist-----') ; print(winL_hist)
            winL_center = winL_center - window_margin + np.argmax(winL_hist)          #; print('window', n, 'LC', winL_center)
            if winL_center < 0:
                winL_center = 0

        winR = bin_img[win_bound_U:win_bound_D, winR_bound_L:winR_bound_R]
        if np.count_nonzero(winR) > min_num_pixel:
            winR_hist = np.sum(winR, axis=0)                                           #; print('-----R hist-----') ; print(winR_hist)
            winR_center = winR_center - window_margin + np.argmax(winR_hist)           #; print('window', n, 'RC', winR_center)
            if winR_center < 0:
                winR_center = 0

        left_lane.coord.append((winL_center, win_bound_U))                             #; print('window', n, 'LC', winL_center)
        right_lane.coord.append((winR_center, win_bound_U))                            #; print('window', n, 'RC', winR_center)
        center_of_lane.append((int((winL_center + winR_center)/2), int((win_bound_U + win_bound_D)/2)))

        if plot:
            cv2.rectangle(bin_img, (winL_bound_L, win_bound_U), (winL_bound_R, win_bound_D), 255, 3)
            cv2.rectangle(bin_img, (winR_bound_L, win_bound_U), (winR_bound_R, win_bound_D), 255, 3)


    if plot:
        #print('left', left_lane.coord)
        #print('right', right_lane.coord)
        for n in range(n_window):
            cv2.line(bin_img, left_lane.coord[n], left_lane.coord[n+1], 100, 3)
            cv2.line(bin_img, right_lane.coord[n], right_lane.coord[n+1], 100, 3)
        for n in range(n_window-1):
            cv2.line(bin_img, center_of_lane[n], center_of_lane[n+1], 255, 3)

    if np.count_nonzero(left_lane.coord, axis=0)[0] < int(n_window/2) or np.count_nonzero(right_lane.coord, axis=0)[0] < int(n_window/2):
        init_start_x(bin_img, left_lane, right_lane)
    else:
        left_lane.start_x = left_lane.coord[1][0]
        right_lane.start_x = right_lane.coord[1][0]

    left_lane.coord.clear()
    right_lane.coord.clear()
