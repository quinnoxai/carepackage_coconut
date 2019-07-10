#import the library opencv
import cv2
import os
#globbing utility.
import glob
#select the path
#I have provided my path from my local computer, please change it accordingly
path = "C:/Users/ShivamM/Desktop/Blur/*.*"
for bb,file in enumerate (glob.glob(path)):
    print(file)
    a= cv2.imread(file)
    print(a)
    # %%%%%%%%%%%%%%%%%%%%%
    #conversion numpy array into rgb image to show
    c = cv2.blur(a, (20,10))
    cv2.imshow('Color image', c)
    #cv2.imwrite("C:/Users/ShivamM/Desktop/simple_images/44_new_1.jpg",c)
    cv2.imwrite('C:/Users/ShivamM/Desktop/Blur/new_{}.jpg'.format(bb), c)
    #wait for 1 second
    k = cv2.waitKey(1000)
    #destroy the windo
    cv2.destroyAllWindows()

