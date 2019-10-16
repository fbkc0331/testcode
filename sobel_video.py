import cv2
import numpy as np
import myopencv as my

"""
import glob
fname = glob.glob('./img/test*.jpg')

for n in fname:
    img = cv2.imread(n, cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(img, 50, 100)
    cv2.imshow(n, canny)
"""

def col_hist(img):
    h, w = img.shape[:2]
    hist = np.zeros((1, w), np.int32)
    print(hist)

    for col in range(w):
        for row in range(h):
            if img[row][col]:
                hist[0][col] = hist[0][col] + 1

    plot = cv2.bitwise_not(np.zeros((h, w), np.uint8))

    for x in range(0, w):
        start = (x, h-1)
        end = (x, h-hist[0, x]-1)
        cv2.line(plot, start, end, [0,0,0], 1)

    cv2.imshow('histogram', plot)

#--------------------------------------------------------------------------------------------------------#
#img = cv2.imread('./img/test1.jpg', cv2.IMREAD_UNCHANGED)
#b, g, r = cv2.split(img)
#cropped_img = my.cropped_img(b, 0.65, 0.93, 0, 1)
#cv2.imshow('cropped_img', cropped_img)

th_x = (35, 100)
th_y = (35, 100)
th_mag = (30, 255)
th_dir = (0.7, 1.3)
#--------------------------------------------------------------------------------------------------------#
"""
src_pts = np.array([[260, 660],
                    [555, 440],
                    [715, 440],
                    [1140, 660]], np.float32)

dst_pts = np.array([[10, 900],
                    [10, 10],
                    [660, 10],
                    [660, 900]], np.float32)
"""
crop = cv2.imread('./img/cropped.jpg')
h, w = crop.shape[:2]

src_pts = np.array([[215, h-1],
                    [520, 0],
                    [715, 0],
                    [1055, h-1]], np.float32)

dst_pts = np.array([[50, 900],
                    [50, 0],
                    [700, 0],
                    [700, 900]], np.float32)
"""
bird = my.bird_eyed_view(crop, src_pts, dst_pts, (660, 900))
cv2.imshow('bird', bird)
"""
cap = cv2.VideoCapture('./img/project_video.mp4')
if cap.isOpened() is not True:
    print('erorr')
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        #cv2.imshow('original', frame)
        b, g, r = cv2.split(frame)
        cropped_img = my.cropped_img(r, 0.65, 0.93, 0, 1)
        #cv2.imshow('cropped', cropped_img)

        sobel_comb = my.sobel_combine(cropped_img, th_x, th_y, th_mag, th_dir)
        cv2.imshow('sobel_comb', sobel_comb)

        bird = my.bird_eyed_view(sobel_comb, src_pts, dst_pts, (800, 900))
        cv2.imshow('bird', bird)

        cropped_bird = my.cropped_img(bird, 0.95, 1, 0, 1)
        cv2.imshow('crop', cropped_bird)

        col_hist(cropped_bird)

        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    else:
        print('error')
        break

#bin_sobel_x = my.bin_sobel_x(cropped_img, th_x)
#bin_sobel_y = my.bin_sobel_y(cropped_img, th_y)
#bin_sobel_mag = my.bin_sobel_mag(cropped_img, th_mag)
#bin_sobel_dir = my.bin_sobel_dir(cropped_img, th_dir)
#sobel_comb = my.sobel_combine(cropped_img, th_x, th_y, th_mag, th_dir)

#cv2.imshow('bin_sobel_x', bin_sobel_x)
#cv2.imshow('bin_sobel_y', bin_sobel_y)
#cv2.imshow('bin_sobel_mag', bin_sobel_mag)
#cv2.imshow('bin_sobel_dir', bin_sobel_dir)
#cv2.imshow('sobel_comb', sobel_comb)

cap.release()

#--------------------------------------------------------------------------------------------------------#
cv2.waitKey(0)
cv2.destroyAllWindows()
#--------------------------------------------------------------------------------------------------------#
