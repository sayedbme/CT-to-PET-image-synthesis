import cv2
import glob
from skimage import io
from ultralytics import YOLO

model = YOLO(".../lung_best.pt")

results = model.predict(source=".../Images/", conf = 0.10)

Bounding_Box = results[0].boxes.xyxy.numpy().astype("uint8") 

image_names = glob.glob(".../Images/*.png")
mask_names = glob.glob(".../Masks/*.png")


#only for showing one results
#previous-masks instead of data; where xy gives coordinates value
import matplotlib.pyplot as plt
img1 = (results[5].masks.data[0].numpy()).astype("uint8")
img2 = (results[5].masks.data[1].numpy()).astype("uint8")
#add two lung 
img = img1 + img2
plt.imshow(img)
plt.show()


##############################################################
##############################################################
detc = []*0
for i in range(0, len(image_names)):
    
    Bounding_Box = results[i].boxes.xyxy.numpy().astype("uint8")
    p,q = Bounding_Box.shape
   
    imrd = io.imread(image_names[i]) #must be io.imread
    imrd = cv2.resize(imrd, (640, 640))
    
    mard = io.imread(mask_names[i]) #must be io.imread
    mard = cv2.resize(mard, (640, 640))
    io.imsave(mask_names[i], mard) # Replace with exixting mask

    
    # Handle both grayscale and color images
    # for color image h, w, c; if it is BW image the h, w 
    if len(imrd.shape) == 2:
        h, w = imrd.shape
    else:
        h, w, c = imrd.shape
    
    
    for j in range(0, h):
            
        if p != 2:
                break
            
        a1 = (results[i].masks.data[0].numpy()).astype("uint8")   
        b1 = (results[i].masks.data[1].numpy()).astype("uint8")
        detc = a1 + b1
        detc = cv2.resize(detc, (640, 640))
            
        for k in range(0, w):            
            if detc[j,k] == 1:        
                imrd[j,k] = imrd[j,k]            
            else:          
                imrd[j,k] = 0 
                
    io.imsave(image_names[i], imrd) # Replace with exixting mask
















