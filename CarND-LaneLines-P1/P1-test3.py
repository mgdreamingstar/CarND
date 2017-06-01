import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_x = np.array([])
    left_y = np.array([])
    right_x = np.array([])
    right_y = np.array([])
    left_start = 0
    left_end = 0
    right_start = 0
    right_end = 0
    slope = 0

    for line in lines:
        # slope = 0
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / (x2-x1)
            if (slope > 0):
                left_x = np.append(left_x,[x1,x2])
                left_y = np.append(left_y,[y1,y2])
            elif (slope < 0):
                right_x = np.append(right_x,[x1,x2])
                right_y = np.append(right_y,[y1,y2])
    left_line = np.polyfit(left_x,left_y,1)
    right_line = np.polyfit(right_x,right_y,1)

    left_y_start = img.shape[0]
    left_x_start = (left_y_start - left_line[1]) / left_line[0]
    left_y_end = 330
    left_x_end = (left_y_end - left_line[1]) / left_line[0]

    right_y_start = img.shape[0]
    right_x_start = (right_y_start - right_line[1]) / right_line[0]
    right_y_end = 330
    right_x_end = (right_y_end - right_line[1]) / right_line[0]

    cv2.line(img, (int(left_x_start), int(left_y_start)), (int(left_x_end), int(left_y_end)), color, thickness)
    cv2.line(img, (int(right_x_start), int(right_y_start)), (int(right_x_end), int(right_y_end)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

image_names = os.listdir("test_images/")

for names in image_names:
    image_dir = 'test_images\\' + names
    # import image
    image = mpimg.imread(image_dir)

    # gray scale
    gray = grayscale(image) # *image* to *gray*

    # gaussian blur
    kernel = 5
    blur_gray = gaussian_blur(gray, kernel) # *gray* to *blur_gray*

    # canny edge detection
    low_t = 50
    high_t = 150
    edges = canny(blur_gray, low_t, high_t) # *blur_gray* to *edges*

    # region of interest, don't need mask.
    imshape = image.shape
    y = imshape[0]
    x = imshape[1]
    vertices = np.array([[(0,y),(450, 330), (550, 330), (x,y)]], dtype=np.int32)
    region = region_of_interest(edges,vertices) # *edge* to *region*

    # hough transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 # minimum number of pixels making up a line
    max_line_gap = 20 # maximum gap in pixels between connectable line segments

    line_image = hough_lines(region, rho, theta, threshold, min_line_len, max_line_gap)

    weigh_img = weighted_img(line_image, image, 0.8, 1)
    image_out_dir = 'test_images_output\\' + names
    mpimg.imsave(image_out_dir,weigh_img)
