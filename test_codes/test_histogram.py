import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from LaneDetector import LaneDetector


import os
import os
os.chdir('..')
image = plt.imread('test_images/test5.jpg')
plt.imshow(image)
plt.show()
laneDetector = LaneDetector(debug=True)
output = laneDetector.process_image(image)
image = plt.imread('test_images/test6.jpg')
output = laneDetector.process_image(image)
import os
from moviepy.editor import VideoFileClip

# white_output = 'project_video_out.mp4'  # New video
# os.remove(white_output)
# clip1 = VideoFileClip('project_video.mp4')  # .subclip(21.00,25.00) # project video
# # clip = VideoFileClip("myHolidays.mp4", audio=True).subclip(50,60)
# white_clip = clip1.fl_image(laneDetector.process_image)  # NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# import glob
# import time
# images = glob.glob('test_images/*.jpg')
#
# for image_path in images:
#     image = mpimg.imread(image_path)
#
#     t = time.time()
#     output = laneDetector.process_image(image)
#     t2 = time.time()
#     print(round(t2 - t, 2), 'procesing time ')
#
#     # t = time.time()
#     # YCrCb_result = train_test_model(ycrcb_parameters, do_training=False, model_path=YCrCb_model_path, image=image)
#     # t2 = time.time()
#     # print(round(t2 - t, 2), 'procesing time for YCrCb.')
#
#     fig = plt.figure()
#     plt.subplot(121)
#     plt.imshow(image)
#     plt.title('Original Image .')
#     plt.subplot(122)
#     plt.imshow(output)
#     plt.title('Lane.')
#     fig.tight_layout()
#     plt.show()

