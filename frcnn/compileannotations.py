from xml.etree import ElementTree
import os
import numpy as np
import math
import cv2
from optparse import OptionParser

commands = OptionParser()
commands.add_option("--bg",dest="includeBackground", help="Choose to include background data.",action="store_true", default=False)

(options, args) = commands.parse_args()

# Create Annotation File
f = open("annotation.txt", "w+")

classes = "classes/"
annotations = "annotations/"
classList = []
for i in os.listdir(classes):
    if i.lower() != 'bg':
        classList.append(i.lower())

class_count = np.zeros(len(classList),dtype=int).tolist()
for i in os.listdir(classes):
    for j in os.listdir(classes+i):
        if i.lower() != 'bg':
            class_count[classList.index(i.lower())] += 1
# max data count
duplicationfactor = 0
offset = 0

duplicates = np.random.choice(min(class_count), size=math.floor(min(class_count)*duplicationfactor))
validation_data = np.random.choice(min(class_count), size=math.floor(min(class_count)*0.08))
test_data = np.random.choice(len(validation_data), size=math.floor(len(validation_data)*0.3))

i = np.zeros(len(classList),dtype=int).tolist()
training_count = np.zeros(len(classList),dtype=int).tolist()
test_count = np.zeros(len(classList),dtype=int).tolist()
validation_count = np.zeros(len(classList),dtype=int).tolist()
for file in os.listdir(annotations):
    if file[(len(file)-3):] != 'xml':
        continue
    xml = ElementTree.parse(annotations + file)
    annot_count = 0
    for object in xml.findall("object"):
        if(object):
            className = object.find('name')
            file_name = xml.find("filename")
            imagePath = classes + className.text + "/" + file_name.text 
            for o in xml.findall("object"):
                # find true image path
                if not os.path.exists(imagePath):                
                    imagePath = classes + o.find('name').text + "/" + file_name.text 
                else:
                    break
            if not os.path.exists(imagePath):                
                continue
            # annotaitons need to be presennted as floating point numbers
            # size = xml.find("size")
            # width = ElementTree.tostringlist(size.find("width"),encoding='unicode')[2]
            # height = ElementTree.tostringlist(size.find("height"),encoding='unicode')[2]
            
            # EXTRACT object details
            setType = ""
            isDuplicate = False
            for k in range(len(validation_data)):
                if i[classList.index(className.text.lower())]  == k + offset:
                    setType = "TRAINVAL"
                else:
                    if setType!="TRAINVAL":
                        setType = "TRAINVAL"
            for k in range(len(test_data)):
                if i[classList.index(className.text.lower())]  == k + offset:
                    setType = "TEST"
            isDuplicate = False 
            for k in range(len(duplicates)):
                if i[classList.index(className.text.lower())]  == k + offset and setType == "TRAINING":
                    isDuplicate = True
            # full_image_added = False
            # for each annotations
            # get path    
           
            box = object.find("bndbox")
            xmin = ElementTree.tostringlist(box.find("xmin"), encoding='unicode')[2]
            ymin = ElementTree.tostringlist(box.find("ymin"), encoding='unicode')[2]
            xmax = ElementTree.tostringlist(box.find("xmax"), encoding='unicode')[2]
            ymax = ElementTree.tostringlist(box.find("ymax"), encoding='unicode')[2]
            
            # convert values to float from 0 - 1
            xminValue = int(xmin)
            yminValue = int(ymin)
            xmaxValue = int(xmax)
            ymaxValue = int(ymax)
            # Add line to annot
            f.write(imagePath + ',' + str(xminValue) + ',' + str(yminValue) + ',' + str (xmaxValue) + ',' + str(ymaxValue) + ',' +  className.text + ',' + setType +'\n')
            if annot_count == 0:
                i[classList.index(className.text.lower())] += 1   
                if setType == "TRAINVAL":
                    training_count[classList.index(className.text.lower())] += 1
                elif setType == "TEST":
                    test_count[classList.index(className.text.lower())] += 1
                elif setType == "TRAINVAL":
                    validation_count[classList.index(className.text.lower())] += 1
            # if(not full_image_added):
            #    full_image = [setType,imagePath, className.text, "0","0","","","1","1","",""]
            #    csvfile_writer.writerow(full_image)
            #    full_image_added = True
            if isDuplicate:
                f.write(imagePath + ',' + str(xminValue) + ',' + str(yminValue) + ',' + str (xmaxValue) + ',' + str(ymaxValue) + ',' +  className.text + ',' + setType + '\n')
                if annot_count == 0:
                    training_count[classList.index(className.text.lower())] += 1
                    i[classList.index(className.text.lower())] += 1    
            annot_count += 1

total = 0
for i in range(len(classList)):
    print(classList[i] + ":\n\t{TrainVal =" + str(training_count[i] + validation_count[i]) + "}")
    print("\t{Test =" + str( test_count[i]) + "}")
    total += training_count[i] + validation_count[i]

print("Total Images = "+  str(total))
bgCount = 0
if options.includeBackground:
    for file in os.listdir(classes + "bg"):
        img = cv2.imread(classes+ "bg/" + file)
        (h, w, _) = img.shape
        f.write(classes + "bg/" + file + ',' + str(1) + ',' + str(1) + ',' + str (w-1) + ',' + str(h-1) + ',' +  "bg"+ ","+ "TRAINVAL "+ '\n')
        bgCount+=1
print("Background Images = " + str (bgCount))
print("Done.")

f.close()
