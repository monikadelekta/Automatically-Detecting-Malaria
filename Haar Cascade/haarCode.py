#!/usr/bin/env python
"""
HAAR DETECTOR CODE FOR AUTOMATIC MALARIA DETECTION
WRITTEN BY MONIKA DELEKTA
"""

import cv2, os
from prettytable import PrettyTable

"""
CLASSIFY
"""
def vChannel(im):
    """
    The image is converted into the HSV colour space and then split into its
    3 channels h, s and v. The v channel is returned to use for classification.
    """
    HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(HSV)
    return v

def classify(classifier, im, im2, scaleFactor, minNeigbours, imNum):
    """
    Take the V channel image and apply the classifier to it, if the
    square to be drawn is larger than 15x15 it is not added to the image
    as this is just some over detection occurring.
    """
    v = vChannel(im)
    cells = classifier.detectMultiScale(v, scaleFactor, minNeigbours)
    rect = 0
    for (x, y, w, h) in cells:
        if (w != 15) & (h != 15): rect += 1
        if (w == 15) & (h == 15): cv2.rectangle(im2, (x, y), (x + w + 3, y + h +3), (0, 0, 255), 2)
    return im2, cells

def calc(cell_count, cell_type, TP, TN, FP, FN):
    """
    If cell_count is 0 it means it did not detect an infection, if it is greater than 0,
    this means it found an infection. This section of code determines if the detection was correct
    or not.
    """
    if len(cell_count) == 0 and cell_type == "uninfected": TN +=1
    if len(cell_count) > 0 and cell_type == "infected": TP += 1
    if len(cell_count) > 0 and cell_type == "uninfected": FP += 1
    if len(cell_count) == 0 and cell_type == "infected": FN += 1
    return FN, TP, TN, FP

def determine_values(TP, TN, FP, FN):
    """
    Calculate the accuracy, precision, recall and f-measure results
    for this detection algorithm.
    """
    accuracy = 0
    precision = 0
    recall = 0
    f_meas = 0

    if (TN + TP + FN + FP) != 0: accuracy = (TN +TP)/ (TN + TP + FN + FP)
    if (TP + FP) != 0: precision = TP / (TP + FP)
    if (TP +FN) != 0: recall = TP / (TP + FN)
    if (precision and recall != 0): f_meas = 2 * ((precision * recall) / precision + recall)
    return accuracy, precision, recall, f_meas

"""
MAIN
"""
TP = 0
TN = 0
FP = 0
FN = 0

print(' """ WELCOME TO THE MALARIA DETECTOR """ ')
print('------------------------------------------ ')
print(' ')

cell_detect = cv2.CascadeClassifier('cascade.xml')
#./cropped_cells
dir_name = input("Please enter the name of the directory that contains the 'infected' and 'uninfected' sub directories: ")

if os.path.exists(dir_name):
    path_cropped = os.path.dirname(__file__) + dir_name
    directory_cropped = os.listdir(path_cropped)
    if len(directory_cropped) == 2:
        if directory_cropped[0] == 'infected' and directory_cropped[1] == 'uninfected':
            for data in directory_cropped:
                files=os.listdir(path_cropped+'/'+data)
                if len(files) == 0:
                    print("There are no files in the subdirectories, please try another folder.")
                    break
                else:
                    print('Your results are being calculated and will be added to a text file for you shortly.')
                    for hdf5_file in files:
                        im = cv2.imread((path_cropped + '/'+ data + '/'+ hdf5_file))
                        im2 = im.copy()
                        class_im, c_count = classify(cell_detect, im, im2, 3.60, 4, hdf5_file)
                        FN, TP, TN, FP = calc(c_count, data, TP, TN, FP, FN)

                    acc, prec, rec, f_meas = determine_values(TP, TN, FP, FN)
                    #show TN, TP, FN, FP counts
                    counts = PrettyTable()
                    counts.field_names = ["TP", "TN", "FP", "FN"]
                    counts.add_row(([TP, TN, FP, FN]))

                    #results table
                    results = PrettyTable()
                    results.field_names = ["Precision", "Recall", "F-measure", "Accuracy"]
                    results.add_row([format(prec, '.2f'), format(rec, '.2f'), format(f_meas, '.2f'), format((acc * 100), '.2f') + "%"])

                    #Send tables to text file
                    new_File = open("Haar_Results.txt", "w+")
                    new_File.write("Results for Haar Cascading on Cropped Images With Background Removed")
                    new_File.write("\n")
                    new_File.write("\n")
                    new_File.write("Counts of TP, TN, FP and FN:")
                    new_File.write("\n")
                    counts_table = counts.get_string()
                    new_File.write(counts_table)
                    new_File.write("\n")
                    new_File.write("\n")
                    new_File.write("Results Calculated: ")
                    new_File.write("\n")
                    final_result = results.get_string()
                    new_File.write(final_result)
                    new_File.write("\n")
                    new_File.write("\n")
                    new_File.close()

                print('Thank you, all of your results have now been added to the text file: ', new_File.name)
                print("--------------------------------------------------------------------------------------")
    else:
        print("The subdirectories do not match the requirement, please ensure your folder contains two sub-folders 'infected' and 'uninfected' and try again.")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
else: print("Sorry, that directory was not found, please try again.")
