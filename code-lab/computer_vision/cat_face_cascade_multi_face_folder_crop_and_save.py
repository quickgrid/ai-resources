"""
Detects cat faces in image, crop and save it in folder
"""


import cv2
import os

print('getcwd:      ', os.getcwd())


detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
z = 0
cur_dir = os.getcwd()


import glob
cv_img = []
for img in glob.glob(cur_dir + "/*.jpg"):
    n= cv2.imread(img)


    #cv_img.append(n).
    #cv2.imshow("Cat Faces", n)
    #cv2.waitKey(0)
    #print(img)


    # load the input image and convert it to grayscale
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=1, minSize=(30, 30))



    # loop over the cat faces and draw a rectangle surrounding each
    for (i, (x, y, w, h)) in enumerate(rects):

        image_out = image[y:y+h, x:x+w]


        #print("output_data\\" + str(z) + ".jpg")
        cv2.imwrite("output_data\\" + str(z) + ".jpg", image_out)



        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


    ## increment image number
    z += 1

    print("processed: " + img)


    # show the detected cat faces, uncomment below to show bounding boxes
    #cv2.imshow("Cat Faces", image)
    #cv2.waitKey(0)
    #cv2.waitKey(1)


cv2.destroyAllWindows()

