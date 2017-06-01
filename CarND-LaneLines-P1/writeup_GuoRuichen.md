# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./Pics_writeup/region.png "Region of Interest"
[image3]: ./Pics_writeup/blur.png "Gaussian Blur"
[image4]: ./Pics_writeup/canny.png "Canny Edge Detection"
[image5]: ./Pics_writeup/output.png "Output"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

First, I converted the images to grayscale.

![alt text][image1]

Then I made the region to contain the lane lines.

![alt text][image2]

Thirdly, I blurred the image with Gaussian blur.

![alt text][image3]

Fourthly, I used the Canny function to detect the edge.

![alt text][image4]

Finally, after Hough transformation to detect the lines, I plotted the lines with the original image.

![alt text][image5]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by `numpy.polyfit()` function. With the lines detected by Hough function, I computed the slope to verify whether the line is belong to the left lane or the right lane, then gather them within some arrays to fit lines with np.polyfit(). Given the y coordinate of starting and ending points of lane lines, it's easy to compute the x coordinate of the points with the line functions. With the points, I can plot the lines with cv2.line() function.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane lins are not straight lines. Because the Hough transformation in my project can only detect straight lines.

Another shortcoming could be different position of lane lines on the pictures and videos. Because the region of interest of mine are set to the low center of the pictures and videos, if the lane lines changed its position, this project may not catch them. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to make the Hough transformation can detect arc and circle either. For now, I haven't complete the challenge part of this project.

Another potential improvement could be to change the parameters of Gaussian blur, Canny detection and Hough transformation. 
