# Automatically-Detecting-Malaria
Repo to automatically detect malaria in blood sample images, using Python.

The below is a summary from my final dissertation which is loaded into the GIT repository.

## Purpose & Abstract

Malaria is spread by a parasite carried in the female mosquito, which is passed to a human when the mosquito bites them. Areas facing poverty are the most affected by Malaria, this is because there is a lack of proper healthcare. It is too expensive to pay laboratory assistants to diagnose blood samples for Malaria. To save costs, these countries simply provide anti-malarial medication to anyone showing any signs or symptoms of Malaria under clinical diagnosis. This, unfortunately, creates a drug resistance making the anti-malarial drugs ineffective. The focus of this project is to produce software that can automatically diagnose Malaria in images of Giemsa stained erythrocytes under a microscope.

The solution was to produce three different versions, this allowed using different techniques to find the most appropriate solution. The first is programmed to take the image and remove sections of it using histogram analysis, thresholding and morphology in stages, until it is left with only the dark blue sections in the image that are assumed to contain the parasite. The second program uses a machine learning approach that has never been used for this type of implementation before, this is Haar Cascading. Similar to the face recognition applications it is used for in smartphones, the cascade was trained to detect the infected cells within the images. The third program is a CNN, this is another machine learning approach which was trained to determine if cells are infected or uninfected through the use of feature and class classification.

Evaluation shows that the CNN returns the highest accuracy and f-measure score, closely behind is the Haar Cascade. The image processing through segmentation program returns the lowest accuracy, making it not useful for real life implementation. The Haar Cascade, however, proves to be the best method, as although it has a lower accuracy and f-measure score than the CNN it is a much more cost- effective method to implement in areas facing poverty.

This work was mostly completed for my final dissertation at university, I have since made tweaks to improve the code and accuracy.  