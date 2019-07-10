import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/ShivamM/Desktop/Single_image/203.Photo2.122727.jpg')

dst = cv2.fastNlMeansDenoisingColored(img,None,5,5,1,21)
#dst_new = cv2.fastNlMeansDenoisingColored(dst,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)


# save the denoised image
#cv2.imwrite('44_new.jpg', dst_new)

final_dst=cv2.fastNlMeansDenoisingColored(dst,None,5,10,7,21)
plt.subplot(122),plt.imshow(final_dst)
plt.show()
cv2.imwrite('C:/Users/ShivamM/Desktop/Single_image/203.Photo2.1292727.jpg', dst)
