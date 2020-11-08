# Automatically-Detecting-Malaria
Repo to automatically detect malaria in blood sample images, using Python.

The below is a summary from my final dissertation which is loaded into the GIT repository.

## Purpose & Abstract

Malaria is spread by a parasite carried in the female mosquito, which is passed to a human when the mosquito bites them. Areas facing poverty are the most affected by Malaria, this is because there is a lack of proper healthcare. It is too expensive to pay laboratory assistants to diagnose blood samples for Malaria. To save costs, these countries simply provide anti-malarial medication to anyone showing any signs or symptoms of Malaria under clinical diagnosis. This, unfortunately, creates a drug resistance making the anti-malarial drugs ineffective. The focus of this project is to produce software that can automatically diagnose Malaria in images of Giemsa stained erythrocytes under a microscope.

The solution was to produce three different versions, this allowed using different techniques to find the most appropriate solution. The first is programmed to take the image and remove sections of it using histogram analysis, thresholding and morphology in stages, until it is left with only the dark blue sections in the image that are assumed to contain the parasite. The second program uses a machine learning approach that has never been used for this type of implementation before, this is Haar Cascading. Similar to the face recognition applications it is used for in smartphones, the cascade was trained to detect the infected cells within the images. The third program is a CNN, this is another machine learning approach which was trained to determine if cells are infected or uninfected through the use of feature and class classification.

Evaluation shows that the CNN returns the highest accuracy and f-measure score, closely behind is the Haar Cascade. The image processing through segmentation program returns the lowest accuracy, making it not useful for real life implementation. The Haar Cascade, however, proves to be the best method, as although it has a lower accuracy and f-measure score than the CNN it is a much more cost- effective method to implement in areas facing poverty.

## Understanding the Images

The data used to produce the three proposed solutions has been split into two. There is a total of 100 images that are available for the entire project. The image set was halved so that 50 images are used for the training and development of the three solutions and the other 50 will be used for the testing of the solutions. There are a further 100 images that are identical to the initial 100 images which contain the ground truth labels. An example of the two images can be seen below.

![Example images to be used for project solutions](Images/Image1.png)

The images used have been taken from a microscopic view of a light microscope where the blood samples were first prepared by staining the blood smear with the Giemsa stain, this stain can be seen working in figure 8 where the infected cells in the ground truth image are a darker colour in comparison to the erythrocytes and background.

There is a large number of defects in the blood such as platelets, artefacts and white blood cells that also get stained with the Giemsa stain when the blood smear is prepared. The background artefacts can be seen in below. The solution for this, as it was in the articles found was to remove the background from the images, however, this needed to be done as carefully as possible to preserve the structure of the erythrocytes within the images.

![Background defects](Images/Image2.png)

## Solutions

There are three solutions to this problem that have been compared. Roughly the steps in each are:

1. Image Processing Through Segmentation
    1. Thresholding
    1. Region Labelling
    1. Morphology
    1. Region Labelling
    1. Calculate Image Moments for Cells
    1. Estimate Image Cell Count
    1. Colour Rebalance Image
    1. Identify and Label Infected Cells
    
2. Haar Cascade 
    1. Create Positive & Negative Image Text File
    1. Create .vec File From Positive Images
    1. Train Cascade
    
3. Convolutional Neural Network
    1. Resize Images and Convert to Grayscale
    1. Label Images
    1. Define Model Architecture
    1. Train/Save HDF5 Model

## Results

![Results Table](Images/Image3.png)

## Discussion and Conclusion

Overall, I have successfully been able to produce three programs as intended, each of the three programs are sufficiently different. The first program showed little promise with a very low accuracy and F-measure. As mentioned earlier this was mainly because of the method used to detect the dark sections within the image. This program is therefore discarded from being an option to be used in the real world, as it only offers a 20.47% accuracy and 28% f-measure which is both cases is extremely low. There are several options in which this program could be improved, the first is to create a separate colour rebalancing algorithm for the different ranges of the darkness of the images as done for the morphology stage. 
By doing so the images passed with their background removed could have a different amount of colour alterations applied to them to ensure each specific range only extracts the Giemsa stained artefacts within the images. A second alteration would be to completely remove the colour rebalance algorithm and apply histogram analysis to each cell or section of an image, if the section/cell contains the blue-black colours there will be a higher peak within the histogram and this can be marked as an infection. This program was however proved useful in the determination of removing the background from images and was used for the cropping tools that I also produced for the training and testing of the Haar Cascade and CNN.

Excluding program 1 there is a choice between the Haar Cascade and CNN. Both programs use supervised machine learning to classify the images and both use cropped cells to be trained and to be tested. For the accuracy value, there is only a 3.87% difference between the Haar Cascade and CNN. If accuracy is being considered, which as mentioned before is not the ideal performance measure to use, there is not a lot of difference to determine correctly which is better. This is because possibly one or two variables could have been tweaked during the Haar Cascade training to possibly reach the same value as the CNN if not even higher. In terms of accuracy, the CNN does return the highest value, but either could be considered as an option.

In terms of f-measure the CNN is once again higher than the Haar Cascade program, however this time it is 10% different. The difference is much larger and shows that the CNN allows for more complex situations within images and is able to make better classification decisions. In terms of only evaluating the F-measure result then the CNN by far is the better option to choose. The f-measure score is most likely the better performance measure to be used as it ensures that unbalanced classes are accounted for. In the case of this project during testing there are a lot more cropped single uninfected cells than there are cropped single infected cells, this makes the accuracy value slightly biased as it does not account for this. For this reason, I would recommend using the CNN. With a higher f-measure it means that the CNN can make stronger decisions based on the number of true detections made.

However, to meet the initial goal of my research it was mentioned that poverty was the main issue. The CNN will be much more expensive to implement than the Haar Cascade, as the Haar Cascade is already implemented in a cost-effective way on smartphone cameras on the industrial scale, it also does not require a lot of processing power. This means that the Haar Cascade could be easily implemented into a smartphone. Whereas with the CNN a lot of research would be required to create a program that will support the model on a smartphone, including the requirement to have processing power. The need for less processing power also means that the Haar Cascade is significantly faster than the CNN, which has shown during testing where it took seconds compared to a few minutes for the CNN. As the CNN requires more processing power, it will also be more expensive to implement. For these reasons, the Haar Cascade would be recommended, it may have slightly more FP values than the CNN but it will still detect very well and it will be much cheaper to implement. The only disadvantage of using the Haar Cascade is that eventually if given too many images it will stop learning as the Haar-like features will become the same and will begin to block further learning, so this would need to be monitored.

Compared to the initial plan I have been able to meet all the intended requirements, with the addition of the CNN which was not initially intended to be produced. Alongside this, the production of the cropping tool and evaluation tools for each program were created and within the time allocated to me. I am satisfied with the results returned which could only be improved with more images that could be provided to the training process.

##### END
