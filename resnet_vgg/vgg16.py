import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import time
import os
start_time = time.time()

csvfile = open("VGG16Features.csv", "w")
model = VGG16(weights='imagenet')

paint_dir = './images'
artist_names = pd.read_csv('./artists.csv')
artist_names = artist_names[artist_names['paintings'] >= 200].reset_index()

artist_file = artist_names['name'].str.replace(' ', '_').values

for name in artist_file:
    if os.path.exists(os.path.join(paint_dir, name)):
        print("Path exists: ", os.path.join(paint_dir, name))
    else:
        print("Not Found: ", os.path.join(paint_dir, name))

artist_names['name'] = artist_names['name'].str.replace(' ', '_')
print(artist_names.head)

for name in artist_file:
    painter_dir = os.path.join(paint_dir, name)
    print("Path: ", painter_dir)
    
    artist = artist_names[artist_names['name'] == name]
    for image_id in range(1, artist['paintings'].iloc[0]+1):
        Final_VGG16Feature = np.zeros((1,1000), np.uint16)
        file1 = painter_dir + "/" + str(name) + "_" + str(image_id) + ".jpg"
        
        if (os.path.isfile(file1)) == False:
            print("Not Found: ", file1)
            continue
        
        image1 = cv2.imread(file1)
        
        img = cv2.resize(image1, (224, 224), interpolation = cv2.INTER_AREA)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        
        Final_VGG16Feature = np.add(Final_VGG16Feature, features)
        
        print("--- %s seconds ---" % (time.time() - start_time))

        temp = 0;
        for data in Final_VGG16Feature[0, :]:
            if(temp<1000):
                csvfile.write(str(data)+ ",")
                temp = temp + 1
        
        csvfile.write("\n")


csvfile.flush()
csvfile.close()