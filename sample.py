from keras.preprocessing import image
import cv2

img=image.load_img('/Users/pranoyr/PycharmProjects/3d-cnn-keras/video_action_recognition/ActionDataImages/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/1.jpg')
img = image.img_to_array(img)
print(img)


img=cv2.imread('/Users/pranoyr/PycharmProjects/3d-cnn-keras/video_action_recognition/ActionDataImages/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/1.jpg')
print(img)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img)
