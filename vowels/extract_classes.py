import numpy as np


def extract_classes_map(filename):
    class_map = {}
    
    with open(filename,"r") as file:
        lines = file.read().split("\n")
        try:
            for line in lines:
                line = line.split(" ")
                for element in line:
                    if element == "":
                        line.remove(element)
                if str(line[0][0]+line[0][-2]+line[0][-1]) not in class_map:
                    class_map[str(line[0][0]+line[0][-2]+line[0][-1])] = line[1:]
                else:
                    class_map[str(line[0][0]+line[0][-2]+line[0][-1])].append(line[1:])
        except:
            print("endOfFile")

    return class_map

classes_map  = extract_classes_map("data.dat")

for c in classes_map:
    print(c)