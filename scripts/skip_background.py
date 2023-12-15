import numpy as np
import cv2
from PIL import Image

def estimate_background(image, thr_low=80, thr_high=220):

    gray_image = np.mean(image * 255, axis=2)
    # gray_image = cv2.medianBlur(gray_image, 5)

    bkg_msk_high = np.where(gray_image > thr_high, np.ones_like(gray_image), np.zeros_like(gray_image))
    bkg_msk_low = np.where(gray_image < thr_low, np.ones_like(gray_image), np.zeros_like(gray_image))

    bkg_msk = np.bitwise_or(bkg_msk_low.astype(np.uint8), bkg_msk_high.astype(np.uint8))
    bkg_msk = cv2.medianBlur(bkg_msk, 5)
    kernel = np.ones((19, 19), np.uint8)
    bkg_msk = cv2.dilate(bkg_msk, kernel)

    bkg_msk = bkg_msk.astype(np.float32)
    return bkg_msk

image = cv2.imread('./088.png')  # sample
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
bkg_msk = estimate_background(image)
mask_img = Image.fromarray((bkg_msk * 255).astype(np.uint8), mode='L')
mask_img.save('mask.png')
# # print(bkg_msk.shape)

# mask = np.zeros(image.shape[:2], np.uint8)

# # Set the foreground and background colors as 1 and 3 respectively
# bgdModel = np.zeros((1,65), np.float64)
# fgdModel = np.zeros((1,65), np.float64)
# rect = (0, 0, 1000, 1000)  # (x,y,w,h) rectangle defining the foreground object

# # Run the GrabCut algorithm for 5 iterations to refine the mask
# cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

# # Create a binary mask where 0 corresponds to background and 1 to foreground
# mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# # Apply the binary mask to the original image to extract the foreground object
# foreground = cv2.bitwise_and(image, image, mask=mask)

# # Apply the inverse of the binary mask to the original image to extract the background
# # background = cv2.bitwise_and(image, image, mask=1-mask)
# cv2.imwrite('foreground.png', foreground)
# # mask_img = Image.fromarray((background * 255).astype(np.uint8), mode='L')
# # mask_img.save('mask.png')

###################################################

# img = cv2.imread('./013.png')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Threshold the image to create a binary mask
# ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# # Find the contours of the foreground objects
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Make sure that at least one contour was found
# # if len(contours) > 0:
# # Find the largest contour (which should be the foreground object)
# largest_contour = max(contours, key=cv2.contourArea)

# # Create a mask of the foreground object
# mask = np.zeros_like(gray)
# cv2.drawContours(mask, [largest_contour], 0, 255, -1)

# # Apply the mask to the original image
# result = cv2.bitwise_and(img, img, mask=mask)

# # Display the result
# cv2.imwrite('foreground.png', result)