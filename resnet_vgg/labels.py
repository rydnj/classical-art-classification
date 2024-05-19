import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvfile = open("label.csv", "w")
artist_names = pd.read_csv('./artists.csv')
artist_names = artist_names[artist_names['paintings'] >= 200].reset_index()
paint_dir = './images'
artist_file = artist_names['name'].str.replace(' ', '_').values
artist_names['name'] = artist_names['name'].str.replace(' ', '_')

for name in artist_file:
    if os.path.exists(os.path.join(paint_dir, name)):
        painter_dir = os.path.join(paint_dir, name)
        print("Path: ", painter_dir)
        artist = artist_names[artist_names['name'] == name]
        for image_id in range(1, artist['paintings'].iloc[0]+1):
            file1 = painter_dir + "/" + str(name) + "_" + str(image_id) + ".jpg"
            
            if (os.path.isfile(file1)) == False:
                print("Not Found: ", file1)
                continue

            print("Writing label: ", file1)

            csvfile.write(str(artist['id'].iloc[0]))
            csvfile.write("\n")


    else:
        print("Not Found: ", os.path.join(paint_dir, name))

csvfile.flush()
csvfile.close()