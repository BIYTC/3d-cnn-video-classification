import os 
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='./video_action_recognition', help='path to dataset')
# opt = parser.parse_args()
image_folder="ActionDataImages"
abs_path=os.path.dirname(os.path.abspath(image_folder))

# cleans .txt files
open(os.path.join(abs_path,'labels.txt'),'w').write("")
open(os.path.join(abs_path,'train.txt'),'w').write("")
open(os.path.join(abs_path,'val.txt'),'w').write("")

# updates labels.txt 
# fight 0
# noFight 1
labels=os.listdir(os.path.join(abs_path,image_folder,'train'))
for i,label in enumerate(labels):
    with open(os.path.join(abs_path,'labels.txt'),'a') as f:
        f.write(str(i)+" "+label)
        f.write('\n')

# loading mapping...
dict_labels={}
a=open(os.path.join(abs_path,'labels.txt'),'r').read()
c=a.split('\n')
for i in c[:len(c)-1]:
    dict_labels.update({i.split(' ')[1]:i.split(' ')[0]})

# generating train.txt 
for i,label in enumerate(labels):
    vid_names=os.listdir(os.path.join(abs_path,image_folder,'train',label))
    for video_name in vid_names:
        with open(os.path.join(abs_path,'train.txt'),'a') as f:
            f.write(os.path.join(abs_path,image_folder,'train',label,video_name)+ " " + dict_labels[label])
            f.write('\n')

# generating train.txt 
for i,label in enumerate(labels):
    video_names=os.listdir(os.path.join(abs_path,image_folder,'val',label))
    for video_name in video_names:
        with open(os.path.join(abs_path,'val.txt'),'a') as f:
            f.write(os.path.join(abs_path,image_folder,'val',label,video_name)+ " " + dict_labels[label])
            f.write('\n')

