import pyscreenshot as ImageGrab
import pygetwindow as gw
import numpy as np
import easyocr
import cv2


# need to run only once to load model into memory
print("Loading Model")
reader = easyocr.Reader(['ch_sim','en'], gpu = True)


# Get by either window title, active window or window directly
print(gw.getAllTitles())
print(gw.getActiveWindow())
print(gw.getAllWindows())


# Provide desired window title manually or with title index of getAllTitles
tmp = gw.getWindowsWithTitle('untitled1 – test2.py PyCharm')


# Print window location, title
print(int(np.abs(tmp[0].left)), int(np.abs(tmp[0].top)), int(np.abs(tmp[0].right)), int(np.abs(tmp[0].bottom)))
print(tmp[0])


# grab fullscreen
#im = ImageGrab.grab()


# grab certain portion of selected window
print("Grabbing Window")
im = ImageGrab.grab(bbox=(int(np.abs(tmp[0].left)), int(np.abs(tmp[0].top)), int(np.abs(tmp[0].right)), int(np.abs(tmp[0].bottom))))  # X1,Y1,X2,Y2



# save image file and process that image
#im.save("window1.jpg")
#result = reader.readtext('window1.jpg')



# Resize the image to certain percent of original image
print("Resizing Image")
im_np = np.array(im)
scale_percent = 50
width = int(im_np.shape[1] * scale_percent / 100)
height = int(im_np.shape[0] * scale_percent / 100)
dim = (width, height)
im_resized = cv2.resize(im_np, dim, interpolation = cv2.INTER_AREA)



# process the grabbed image directly as numpy array
print("Processing with OCR")
result = reader.readtext(np.array(im_resized))
print(result)



# For each detected text draw bounding box in image and print text, location
i = 0
for r in result:
    print("################################")
    print(r[0][0], r[0][1], r[0][2], r[0][3])
    print("FOUND:")
    print(r[1])
    print("################################")

    x1=min(r[0][0][0], r[0][1][0], r[0][2][0], r[0][3][0])
    x2=max(r[0][0][0], r[0][1][0], r[0][2][0], r[0][3][0])
    y1=min(r[0][0][1], r[0][1][1], r[0][2][1], r[0][3][1])
    y2=max(r[0][0][1], r[0][1][1], r[0][2][1], r[0][3][1])

    image = cv2.rectangle(im_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)

    cv2.imshow("window_name", image)
    cv2.waitKey(0)
    cv2.imwrite("images/img_" + str(i) + ".jpg", image)
    i += 1
