#!/usr/bin/env python
"""
EVALUATION TOOL FOR TESTING AUTOMATIC MALARIA DETECTION CODE
WRITTEN BY MONIKA DELEKTAs
"""
import cv2, numpy as np, os
import pandas as pd
from skimage import measure
from prettytable import PrettyTable

def infectedMask(im):
    """
    Creates the mask to find the 'red' sections of the detected image.
    """
    im =cv2.resize(im, (700,555))
    lowerRange = np.array([0,0,190])
    upperRange = np.array([40,50,255])
    shape = cv2.inRange(im, lowerRange, upperRange)
    return shape

def groundTruthLabel(im):
    """
    Creates the mask to find the ground truth labels in the images. This is done by first taking
    the range to find the black squares, then the defects caught during this process are removed.
    """
    lowerRange = np.array([0,0,0])
    upperRange = np.array([60,60,60])
    ground_truth = cv2.inRange(im, lowerRange, upperRange)

    #take the mask and remove any caught white defects from the image
    shape = ground_truth.shape
    defect_find = measure.label(ground_truth, neighbors=4)
    mask = np.zeros(shape, dtype="uint8")
    for i in pd.DataFrame(defect_find):
        if (i != 0):
            mask_located = np.zeros(shape, dtype="uint8")
            mask_located[defect_find == i] = 255
            pixel_count = cv2.countNonZero(mask_located)
            if pixel_count > 350: mask = mask_located + mask
        else: continue
    return mask

def numberOfLabels(im):
    im2 = im.copy()
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    ima, cnts, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im2, cnts, -1, (0, 0, 255), 2)
    single = 0
    double = 0
    triple = 0
    #get single squares not overlapping etc
    # 46 causes issues & 3
    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        #single squares
        if ((w >= 41)&(w <= 47))& ((h >= 41)&(h <= 55)): single+=1
        #double squares
        if ((w >= 44) & (w <= 47)) & ((h >= 70) & (h <= 90)): double += 2
        if ((w >= 48)&(w < 75))& ((h >= 40)&(h <= 90)): double += 2
        if ((w >= 75) & (w < 95)) & ((h >= 40) & (h < 80)): double += 2
        #triple squares
        if ((w >= 80) & (w <= 140)) & ((h >= 80) & (h <= 140)): triple += 3
        if ((w >= 70) & (w <= 75)) & ((h >= 90) & (h <= 100)): triple += 3
        if ((w >= 100) & (w <= 140)) & ((h >= 50) & (h < 85)): triple += 3

    total = single+double+triple
    print("Total: ", total)
    return total

def orImages(groundTruthIm, evalIm):
    kernel = np.ones((2,2), np.uint8)
    groundTruthIm = cv2.erode(groundTruthIm, kernel, iterations=1)
    fullIm = groundTruthIm | evalIm
    #cv2.imshow("OR", fullIm)
    return fullIm

def contouring(orIm):
    orIm2 = orIm.copy()
    orIm2 = cv2.cvtColor(orIm2, cv2.COLOR_GRAY2BGR)
    im2, cnts, hier = cv2.findContours(orIm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(orIm2, cnts, -1, (0, 0, 255), 2)
    #cv2.imshow("Contouring", orIm2)
    return cnts, hier

def infectionMatches(cnts, hier, total_infected):
    true_positive_count = 0
    false_positive_count = 0
    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])

        #  0       1          2          3
        #[Next, Previous, First Child, Parent]
        #print(hier[0, i, 0], hier[0, i, 1], hier[0, i, 2], hier[0, i, 3])
        #print(w, h)
        if ((w == 5) | (w==4) & (h ==5)|(h == 4)) | ((w == 7) | (w==8) & (h ==8)|(h == 7)):
            if (hier[0, i, 3] == -1): false_positive_count += 1
            #print(hier[0, i, 0], hier[0, i, 1], hier[0, i, 2], hier[0, i, 3])
        if ((w >= 34) &(w<85)) & ((h >= 34) & (h < 85)):
            if ((hier[0, i, 2]) != -1) & ((hier[0,i,3])!= -1): true_positive_count += 1

    false_negative_count = total_infected - true_positive_count
    accuracy_percentage = (true_positive_count/total_infected) * 100
    print("True Positive Count: ", true_positive_count)
    print("False Negative Count: ", false_negative_count)
    print("False Positive Count: ", false_positive_count)
    return true_positive_count, false_positive_count, false_negative_count

"""
Calculations
"""

def accuracy(TP, FP, FN):
    TN = 0
    if ((TP+TN+FP+FN) != 0):
        accuracy_value =((TP+TN)/(TP+TN+FP+FN))* 100
        accuracy_value2 = format(accuracy_value, '.2f')
        print("Accuracy of image: ", accuracy_value2+'%')
    else: return 0.00
    return accuracy_value

def precision_recall(TP, FP, FN):
    precision = 0.00
    recall = 0.00
    if (TP+FP != 0):
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    return precision, recall

def f_measure(precision, recall):
    if (precision and recall != 0): F = 2*((precision*recall)/precision+recall)
    else: return 0.00
    print("F-measure of image: ", format(F, '.2f'))
    return F

x = PrettyTable()
x.field_names = ["File", "Precision", "Recall", "F-measure", "Accuracy"]

prec_count = 0
rec_count = 0
f_m_count = 0
acc_count = 0
doc_count = 0

def add_Rows(fName, prec, rec, f_m, acc):
    x.add_row([fName, format(prec, '.2f'), format(rec, '.2f'), f_m, acc])

dirHaar = "./savedImProc"
dirLabelled = "./ground"
dirHaarFiles = set(os.listdir(dirHaar))
dirLabelledFiles = set(os.listdir(dirLabelled))
combinedSet = dirHaarFiles & dirLabelledFiles

for shared in combinedSet:
    print("Shared: ", shared)
    doc_count += 1
    haarTestIm = cv2.imread(dirHaar + "/" + shared)
    ground_truth_im = cv2.imread(dirLabelled + "/" + shared)
    infectedMask(haarTestIm)
    groundTruthLabel(ground_truth_im)
    total_infected = numberOfLabels(groundTruthLabel(ground_truth_im))
    orImages(groundTruthLabel(ground_truth_im), infectedMask(haarTestIm))
    contour, hierarchy = contouring(orImages(groundTruthLabel(ground_truth_im), infectedMask(haarTestIm)))
    TP, FP, FN = infectionMatches(contour, hierarchy, total_infected)
    acc = accuracy(TP, FP, FN)
    prec, rec = precision_recall(TP, FP, FN)
    f = f_measure(prec, rec)

    prec_count += prec
    rec_count += rec
    f_m_count += f
    acc_count += acc

    add_Rows(str(shared), prec, rec, format(f, '.2f'), format(acc, '.2f'))

    wait = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if wait == 27:
     cv2.destroyAllWindows()
     break

add_Rows(str("Averages: "), prec_count/doc_count, rec_count/doc_count, format(f_m_count/doc_count, '.2f'), format(acc_count/doc_count, '.2f'))
print(x)
