import cv2
import os

image_folder = 'water-security-of-india/code//images/zone/7'
video_name = 'Zone7.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images=sorted(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
