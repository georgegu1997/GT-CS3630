#!/usr/bin/env python

##############
#### Your name: Qiao Gu
##############

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color, transform
import pickle

import matplotlib.pyplot as plt

class ImageClassifier:

    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        #create one large array of image data
        data = io.concatenate_images(ic)

        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE

        # covert the images to greyscale
        processed_image = color.rgb2gray(data)
        # plt.subplot(321)
        # plt.imshow(processed_image[0])
        # plt.subplot(322)
        # plt.imshow(processed_image[1])

        # use Contrast Limited Adaptive Histogram Equalization (CLAHE).
        for i in range(processed_image.shape[0]):
            processed_image[i] = filters.gaussian(processed_image[i], sigma = 1.5)
            processed_image[i] = exposure.equalize_hist(processed_image[i])

        # plt.subplot(323)
        # plt.imshow(processed_image[0])
        # plt.subplot(324)
        # plt.imshow(processed_image[1])

        # extract the hog of the images
        feature_data = []
        for i in range(processed_image.shape[0]):
            feature_img = feature.hog(processed_image[i],
                orientations=12,
                pixels_per_cell=(32, 32),
                cells_per_block=(5, 5),
                block_norm='L2-Hys')
            feature_data.append(feature_img)

        feature_data = np.array(feature_data)

        # print(feature_data.shape)
        #
        # plt.subplot(325)
        # plt.imshow(hog_image[0])
        # plt.subplot(326)
        # plt.imshow(hog_image[1])
        # plt.show()

        ########################

        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier

        ########################
        ######## YOUR CODE HERE
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

        ########################

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        ########################
        ######## YOUR CODE HERE
        predicted_labels = self.classifer.predict(data)
        ########################

        # Please do not modify the return type below
        return predicted_labels


def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    # print(train_raw.shape)
    # (196, 240, 320, 3)
    # print(train_labels.shape)
    # (196,)
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    # print(test_raw.shape)
    # (40, 240, 320, 3)
    # print(test_labels.shape)
    # (40,)

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTesting results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    pickle.dump(img_clf, open("clf.txt", "wb"))


if __name__ == "__main__":
    main()
