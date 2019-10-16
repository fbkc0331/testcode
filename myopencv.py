import cv2
import numpy as np


#---------------------------------------------------  TRANSFORMATION  ---------------------------------------------------#
def bird_eyed_view(img, src_pts, dst_pts, sz_output):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    persp_img = cv2.warpPerspective(img, M, sz_output)

    return persp_img


#---------------------------------------------------  EDGE DETECTION  ---------------------------------------------------#
# function : sobel edge detection
def sobel_xy(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    #abs_sobel_x = np.absolute(sobel_x)
    #abs_sobel_y = np.absolute(sobel_y)
    #scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    #scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))

    return scaled_sobel_x, scaled_sobel_y


# function : canny edge detection with (n x n) size gaussian kernel
def guassian_canny(img, sz_kernel, low_th, high_th):
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    blur_img = cv2.GaussianBlur(gray_img, (sz_kernel, sz_kernel), 0)
    canny_img = cv2.Canny(blur_img, low_th, high_th)

    return canny_img


# function : get hough line segment
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    return lines


# function : draw lines for given coordinates
def draw_lines(img, lines, color=[0, 0, 255], thickness=1):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            print(x1, y1, x2, y2)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    cv2.imshow('line', line_img)

    overlapped_img = cv2.addWeighted(img, 1, line_img, 1, 0)
    return overlapped_img


#---------------------------------------------------  ETC FUNCTION  ---------------------------------------------------#
# function : get cropped image
def cropped_img(img, h1_ratio, h2_ratio, w1_ratio, w2_ratio):
    if (0<=h1_ratio<=1 and 0<=h2_ratio<=1 and 0<=w1_ratio<=1 and 0<=w2_ratio<=1) is False:
        print('error : choose 0 <= ratio <= 1')
        exit()

    if (h1_ratio < h2_ratio and w1_ratio < w2_ratio) is False:
        print('error : choose h_ratio1 < h_ratio2 , w_ratio1 < w_ratio2')
        exit()

    h, w = img.shape[:2]
    h1 = int(h1_ratio*h) ; h2 = int(h2_ratio*h)
    w1 = int(w1_ratio*w) ; w2 = int(w2_ratio*w)

    if len(img.shape) == 3:
        cropped_img = np.zeros((h2-h1, w2-w1, 3), np.uint8)
    else:
        cropped_img = np.zeros((h2-h1, w2-w1), np.uint8)

    cropped_img[0:h2-h1, 0:w2-w1] = img[h1:h2, w1:w2]
    return cropped_img


# function : get ROI of image. be careful with the order of points
def get_roi(img, poly_coord):           # coord : data - np.array, size - (nx2), type - np.int32
    if len(img.shape) == 3:
        color = (255, 255, 255)
    else:
        color = 255

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly_coord, color)
    roi_img = cv2.bitwise_and(img, mask)

    return roi_img

#-------------------------------------------------#
#              Example (rect window)              #
#-------------------------------------------------#
# h, w = img.shape[:2]                            #
# coord = np.array([[0.25*w, 0.25*h],             #
#                   [0.25*w, 0.75*h],             #
#                   [0.75*w, 0.75*h],             #
#                   [0.75*w, 0.25*h]], np.int32)  #
# roi = my.get_roi(img, [coord])                  #
#-------------------------------------------------#


# function : plot histogram
def plot_hist(img, mask=None, bin=256, x_range=[0, 256], thickness=2, y_max=256*2):
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], mask, [bin], x_range)
        y_scale = y_max/np.max(hist)
        plot = cv2.bitwise_not(np.zeros((y_max, 256*thickness), np.uint8))

        for x in range(0, 256):
            start = (x*thickness, y_max-1)
            end = (x*thickness, int(y_max-hist[x, 0]*y_scale-1))
            cv2.line(plot, start, end, [0,0,0], thickness)

        cv2.imshow('grayscale histogram', plot)

    else:
        hist_b = cv2.calcHist([img], [0], mask, [bin], x_range)
        hist_g = cv2.calcHist([img], [1], mask, [bin], x_range)
        hist_r = cv2.calcHist([img], [2], mask, [bin], x_range)

        max_of_max = max([np.max(hist_b), np.max(hist_g), np.max(hist_r)])
        y_scale = y_max/max_of_max

        plot_b = cv2.bitwise_not(np.zeros((y_max, 256*thickness), np.uint8))
        plot_g = cv2.bitwise_not(np.zeros((y_max, 256*thickness), np.uint8))
        plot_r = cv2.bitwise_not(np.zeros((y_max, 256*thickness), np.uint8))

        for x in range(0, 256):
            start = (x*thickness, y_max-1)

            end_b = (x*thickness, int(y_max-hist_b[x, 0]*y_scale-1))
            end_g = (x*thickness, int(y_max-hist_g[x, 0]*y_scale-1))
            end_r = (x*thickness, int(y_max-hist_r[x, 0]*y_scale-1))

            cv2.line(plot_b, start, end_b, [0,0,0], thickness)
            cv2.line(plot_g, start, end_g, [0,0,0], thickness)
            cv2.line(plot_r, start, end_r, [0,0,0], thickness)

        cv2.imshow('blue histogram', plot_b)
        cv2.imshow('green histogram', plot_g)
        cv2.imshow('red histogram', plot_r)






#---------------------------------------------------  BINARY EDGE WITH THRESHOLD  ---------------------------------------------------#
def bin_sobel_x(img, th_x=(20, 100)):
    abs_sobel_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    scaled_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))

    bin_sobel_x = np.zeros_like(scaled_sobel_x)
    bin_sobel_x[(scaled_sobel_x >= th_x[0]) & (scaled_sobel_x <= th_x[1])] = 255

    return bin_sobel_x


def bin_sobel_y(img, th_y=(20, 100)):
    abs_sobel_y = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))

    bin_sobel_y = np.zeros_like(scaled_sobel_y)
    bin_sobel_y[(scaled_sobel_y >= th_y[0]) & (scaled_sobel_y <= th_y[1])] = 255

    return bin_sobel_y


def bin_sobel_mag(img, th_mag=(200, 255)):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    #scale_factor = np.max(sobel_mag)/255
    #scaled_sobel_mag = (sobel_mag/scale_factor).astype(np.uint8)
    scaled_sobel_mag = np.uint8(255*sobel_mag/np.max(sobel_mag))

    bin_sobel_mag = np.zeros_like(scaled_sobel_mag)
    bin_sobel_mag[(scaled_sobel_mag >= th_mag[0]) & (scaled_sobel_mag <= th_mag[1])] = 255

    return bin_sobel_mag


def bin_sobel_dir(img, th_dir=(0.7, 1.3)):
    abs_sobel_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    abs_sobel_y = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    abs_sobel_dir = np.arctan2(abs_sobel_y, abs_sobel_x)

    bin_sobel_dir = np.zeros_like(abs_sobel_dir)
    bin_sobel_dir[(abs_sobel_dir >= th_dir[0]) & (abs_sobel_dir <= th_dir[1])] = 255

    return bin_sobel_dir.astype(np.uint8)


def sobel_combine(img, th_x, th_y, th_mag, th_dir):
    sobel_x = bin_sobel_x(img, th_x)
    #cv2.imshow('sobel_x', sobelx)
    sobel_y = bin_sobel_y(img, th_y)
    #cv2.imshow('sobel_y', sobely)
    sobel_mag = bin_sobel_mag(img, th_mag)
    #cv2.imshow('sobel_mag', mag_img)
    sobel_dir = bin_sobel_dir(img, th_dir)
    #cv2.imshow('result5', dir_img)

    sobel_comb = np.zeros_like(sobel_dir).astype(np.uint8)
    sobel_comb[((sobel_x > 1) & (sobel_mag > 1) & (sobel_dir > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255

    return sobel_comb


def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
    return binary


def hls_combine(img, th_h, th_l, th_s):
    # convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2]
    R = img[220:rows - 12, 0:cols, 2]
    _, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    #cv2.imshow('red!!!',R)
    H = hls[220:rows - 12, 0:cols, 0]
    L = hls[220:rows - 12, 0:cols, 1]
    S = hls[220:rows - 12, 0:cols, 2]

    h_img = ch_thresh(H, th_h)
    #cv2.imshow('HLS (H) threshold', h_img)
    l_img = ch_thresh(L, th_l)
    #cv2.imshow('HLS (L) threshold', l_img)
    s_img = ch_thresh(S, th_s)
    #cv2.imshow('HLS (S) threshold', s_img)

    # Two cases - lane lines in shadow or not
    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 # | (R > 1)] = 255
    #hls_comb[((s_img > 1) & (h_img > 1)) | (R > 1)] = 255
    return hls_comb

def comb_result(grad, hls):
    """ give different value to distinguish them """
    result = np.zeros_like(hls).astype(np.uint8)
    #result[((grad > 1) | (hls > 1))] = 255
    result[(grad > 1)] = 100
    result[(hls > 1)] = 255

    return result
