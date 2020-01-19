'''

This is the part of https://github.com/githubharald/SimpleHTR with simple modification
See License.
'''

from __future__ import division
from __future__ import print_function


import random
import os
import cv2
import numpy as np

from SamplePreprocessor import preprocessor


class FilePaths:
    """ Filenames and paths to data """
    fnCharList = '../model/charList.txt'
    fnWordCharList = '../model/wordCharList.txt'
    fnCorpus = '../data/corpus.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/testImage1.png'  ## path to recognize the single image


class Sample:
    """ Sample from the dataset """

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    """ Batch containing images and ground truth texts """

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen, load_aug=True):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1] == '/'

        self.dataAugmentation = True # False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open("../data/" + 'lines.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')  ## remove the space and split with ' '
            # assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            #print(fileNameSplit)
            fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +\
                       lineSplit[0] + '.png'

            # GT text are columns starting at 10
            # see the lines.txt and check where the GT text starts, in this case it is 10
            gtText_list = lineSplit[9].split('|')
            gtText = self.truncateLabel(' '.join(gtText_list), maxTextLen)
            chars = chars.union(set(list(gtText)))  ## taking the unique characters present

            # check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            # put sample into list
            self.samples.append(Sample(gtText, fileName))


        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: 95% - 10%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]
        print("Train: {}, Validation: {}".format(len(self.trainSamples), len(self.validationSamples)))
        # put lines into lists
        self.trainLines = [x.gtText for x in self.trainSamples]
        self.validationLines = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 9500

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples) # shuffle the samples in each epoch
        self.samples = self.trainSamples #[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocessor(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)
