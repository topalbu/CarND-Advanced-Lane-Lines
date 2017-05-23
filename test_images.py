import glob
import time
from LaneDetector import LaneDetector
from Lane import Lane
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

images = glob.glob('test_images/*.jpg')

for image_path in images:
    image = mpimg.imread(image_path)
    # create the lane detector object to search for the lanes
    laneDetector = LaneDetector()
    t = time.time()
    output = laneDetector.process_image(image)
    t2 = time.time()
    print(round(t2 - t, 2), 'procesing time ')

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image .')
    plt.subplot(122)
    plt.imshow(output)
    plt.title('Output.')
    fig.tight_layout()
    plt.show()