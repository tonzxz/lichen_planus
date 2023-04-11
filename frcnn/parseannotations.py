import os
files = {}
with open("annotation.txt",'r') as f:
    print("Regetting annoatations from compiled annotations")
    for line in f:
        line_split = line.strip().split(',')
        (filepath,x1,y1,x2,y2,class_name,imageset) = line_split
        filename = os.path.basename(filepath).split('/')[-1].split(".")
        xmlFile = ""
        for i in range(len(filename)-1):
            xmlFile+=filename[i] + "."
        xmlFile+="xml"
        if class_name != "bg":
            if "annotations/" + xmlFile not in files:
                c = open("annotations/" + xmlFile, "w")
                files["annotations/" + xmlFile] = "annotations/" + xmlFile
                c.write("<annotation>\n")
                c.write("\t<folder></folder>\n")
                c.write("\t<filename>{}</filename>\n".format(os.path.basename(filepath).split('/')[-1]))
                c.write("\t<path></path>\n")
                c.write("\t<source>\t<database>Unknown</database></source>\n")
                c.write("\t<size>\t<width></width>\n\t<height></height>\n\t<depth></depth>\n</size>\n")
                c.write("\t<segmented>0</segmented>\n")
            else:
                c = open("annotations/" + xmlFile, "a")
            c.write("\t<object>\n"\
                    +"\t\t<name>{}</name>\n".format(class_name)\
                    +"\t\t<pose>Unspecified</pose>\n"\
                    +"\t\t<truncated>0</truncated>\n"\
                    +"\t\t<difficult>0</difficult>\n"\
                    +"\t\t<bndbox>\n"\
                    +"\t\t\t<xmin>{}</xmin>\n".format(x1)\
                    +"\t\t\t<ymin>{}</ymin>\n".format(y1)\
                    +"\t\t\t<xmax>{}</xmax>\n".format(x2)\
                    +"\t\t\t<ymax>{}</ymax>\n".format(y2)\
                    +"\t\t</bndbox>\n"\
                "\t</object>\n")
            # add annotations
            c.close()
for i in files:
    c = open(i, "a") 
    c.write("</annotation>")
    c.close

print("done.")