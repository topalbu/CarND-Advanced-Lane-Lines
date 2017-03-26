import numpy as np
import cv2
class PerspectiveTransform:
    src = np.float32([(130, 700), (540, 470), (740, 470), (1150, 700)])
    dst = np.float32([(330, 720), (330, 0), (950, 0), (950, 720)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    def get_perpective_transform(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def get_reverse_transform(self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)