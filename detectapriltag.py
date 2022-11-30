#! /usr/bin/env python3

import cv2 as cv
# from dt_apriltags import Detector
import matplotlib.pyplot as plt
import rawpy
import os
import numpy as np
# from `apriltag/build import apriltag.pc
from pupil_apriltags import Detector # this one is updated for AprilTag3

# https://github.com/duckietown/lib-dt-apriltags

# load gray
# filename = 'images/test1.jpg'
# filename = 'images/AlphaTag36_11.jpg'
# filename = 'images/AlphaTag36_11-2.jpg' # croped so just one tag
# filename = 'images/AlphaTag36_11-5.jpg' # editted to remove chromatic aberattion and lens profile
# filename = 'images/DSC_0463.jpg'
# filename = 'images/20221130_195854.jpg'
filename = 'images/20221130_195924.JPG'

img = cv.imread(filename)
# img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

# filename = 'images/AlphaTag36_11.ARW'
# with rawpy.imread(filename) as raw:
#     img = raw.postprocess(use_auto_wb=True)

gray = img
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# downsize image to much smaller scale
scale = 1.0
gray = cv.resize(gray, (0, 0), fx=scale, fy = scale)

# plt.imshow(gray)
# plt.show()

# detection of april tag
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=5.0,
                       quad_sigma=2.0,
                       refine_edges=1,
                       decode_sharpening=0.0,
                       debug=1)
detections = at_detector.detect(gray, 
                          estimate_tag_pose=False, 
                          camera_params=None,
                          tag_size=None) # tag size in meters

# detector = apriltag('tagStandard36h11')
# detections = detector.detect(gray)

print("{} total AprilTags detected".format(len(detections)))
print(detections)

# loop over the AprilTag detection results
thickness=10
fontthickness=10
for tag in detections:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
    # NOTE also rescale back up to original image dimensions

	for idx in range(len(tag.corners)):
		
		cv.line(img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), thickness)

	cv.putText(img, str(tag.tag_id),
		org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
		fontFace=cv.FONT_HERSHEY_SIMPLEX,
		fontScale=4.0,
		color=(0, 0, 255),
  		thickness=fontthickness)


img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()



import code
code.interact(local=dict(globals(), **locals()))