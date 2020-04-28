import numpy as np


def extract_classes_map(filename):
    class_map = {}
    
    with open(filename,"r") as file:
        lines = file.read().split("\n")
        try:
            for line in lines:
                line = line.split(" ")                

                x_i = []
                for element in range(1,len(line)):
                    if line[element] != '':
                        x_i.append(line[element])
                    
                if str(line[0][-2]+line[0][-1]) not in class_map:
                    class_map[str(line[0][-2]+line[0][-1])] = [x_i]
                else:
                    class_map[str(line[0][-2]+line[0][-1])].append(x_i)
        except IndexError:
            print("End of File")


    return class_map


