# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import cv2

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
    return weigh_img

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
