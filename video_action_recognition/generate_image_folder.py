import os 
import cv2

image_folder='ActionDataImages'
video_folder='ActionData'
# detete image folder
os.system("rm -rf "+image_folder)
# create image folder
os.system("mkdir "+image_folder)

for train_val in os.listdir(video_folder):
     # make train/val dir
     os.system("mkdir "+os.path.join(image_folder,train_val))
     for label in os.listdir(os.path.join(video_folder,train_val)):
          # make label dir
          os.system("mkdir "+os.path.join(image_folder,train_val,label))
          for video_name in os.listdir(os.path.join(video_folder,train_val,label)):
               i=0
               # make folder
               os.system("mkdir "+os.path.join(image_folder,train_val,label,video_name.split('.')[0]))
               camera=cv2.VideoCapture(os.path.join(video_folder,train_val,label,video_name))
               # read frames
               while True:
                    ret,img=camera.read()
                    if ret==False:
                         break   
                    img_name=str(i)+'.jpg'
                    print(os.path.join(image_folder,train_val,label,video_name.split('.')[0],img_name))
                    # save frames
                    cv2.imwrite(os.path.join(image_folder,train_val,label,video_name.split('.')[0],img_name),img)
                    i+=1
               
            