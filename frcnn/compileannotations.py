from xml.etree import ElementTree
import os
import numpy as np
import math

# Create Annotation File
f = open("annotation.txt", "w+")

classes = "classes/"
annotations = "annotations/"
classList = []
for i in os.listdir(classes):
    classList.append(i.lower())

class_count = np.zeros(len(classList),dtype=int).tolist()
for i in os.listdir(classes):
    for j in os.listdir(classes+i):
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
    xml = ElementTree.parse(annotations+ file)
    class_folder = xml.find("folder")
    file_name = xml.find("filename")
    file_path = xml.find('path')
    if not os.path.exists(classes+ class_folder.text+ '/'+file_name.text):
        continue
    size = xml.find("size")
    width = ElementTree.tostringlist(size.find("width"),encoding='unicode')[2]
    height = ElementTree.tostringlist(size.find("height"),encoding='unicode')[2]
    # EXTRACT object details
    setType = ""
    isDuplicate = False
    for k in range(len(validation_data)):
        if i[classList.index(class_folder.text.lower())]  == k + offset:
            setType = "VALIDATION"
        else:
            if setType!="VALIDATION":
                setType = "TRAINING"
    for k in range(len(test_data)):
        if i[classList.index(class_folder.text.lower())]  == k + offset:
            setType = "TEST"
    isDuplicate = False 
    for k in range(len(duplicates)):
        if i[classList.index(class_folder.text.lower())]  == k + offset and setType == "TRAINING":
            isDuplicate = True
    # full_image_added = False
    # for each annotations
    annot_count = 0
    for object in xml.findall("object"):
        if(object):
            className = object.find('name')
             # get path    
            imagePath = classes + className.text + "/" + file_name.text 
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
            f.write(imagePath + ',' + str(xminValue) + ',' + str(yminValue) + ',' + str (xmaxValue) + ',' + str(ymaxValue) + ',' +  className.text + '\n')
            if annot_count == 0:
                i[classList.index(class_folder.text.lower())] += 1   
                if setType == "TRAINING":
                    training_count[classList.index(class_folder.text.lower())] += 1
                elif setType == "TEST":
                    test_count[classList.index(class_folder.text.lower())] += 1
                elif setType == "VALIDATION":
                    validation_count[classList.index(class_folder.text.lower())] += 1
            # if(not full_image_added):
            #    full_image = [setType,imagePath, className.text, "0","0","","","1","1","",""]
            #    csvfile_writer.writerow(full_image)
            #    full_image_added = True
            if isDuplicate:
                f.write(imagePath + ',' + str(xminValue) + ',' + str(yminValue) + ',' + str (xmaxValue) + ',' + str(ymaxValue) + ',' +  className.text + '\n')
                if annot_count == 0:
                    training_count[classList.index(class_folder.text.lower())] += 1
                    i[classList.index(class_folder.text.lower())] += 1    
            annot_count += 1

for i in range(len(classList)):
    print(classList[i] + ":\n\t{TRAINING=" + str(training_count[i])+ ", TEST=" +str(test_count[i])+", VALIDATION="+str(validation_count[i])+ "}")
print("Done.")

f.close()