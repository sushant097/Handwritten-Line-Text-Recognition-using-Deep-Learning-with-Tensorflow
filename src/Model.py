from __future__ import division
from __future__ import print_function
import codecs
import sys

import tensorflow as tf

from DataLoader import FilePaths


class DecoderType:
    BestPath = 0
    WordBeamSearch = 1
    BeamSearch = 2


class Model:
    # Model Constants
    batchSize = 10 # 50
    imgSize = (800, 64) 
    maxTextLen = 100 # maximum text length can be reccognized

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0


        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN

        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        # Initialize TensorFlow
        (self.sess, self.saver) = self.setupTF()

        self.training_loss_summary = tf.summary.scalar('loss', self.loss)
        self.writer = tf.summary.FileWriter(
           './logs', self.sess.graph)  # Tensorboard: Create writer
        self.merge = tf.summary.merge([self.training_loss_summary])  # Tensorboard: Merge

    def setupCNN(self):
        """ Create CNN layers and return output of these layers """

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        with tf.name_scope('Conv_Pool_1'):
            kernel = tf.Variable(
                tf.truncated_normal([5, 5, 1, 64], stddev=0.1))
            conv = tf.nn.conv2d(
                cnnIn4d, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Second Layer: Conv (5x5) + Pool (1x2) - Output size: 400 x 16 x 128
        with tf.name_scope('Conv_Pool_2'):
            kernel = tf.Variable(tf.truncated_normal(
                [5, 5, 64, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 8 x 128
        with tf.name_scope('Conv_Pool_BN_3'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 128, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Fourth Layer: Conv (3x3) - Output size: 200 x 8 x 256
        with tf.name_scope('Conv_4'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 128, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)

        # Fifth Layer: Conv (3x3) + Pool(2x2) - Output size: 100 x 4 x 256
        with tf.name_scope('Conv_Pool_5'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 256, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                learelu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Sixth Layer: Conv (3x3) + Pool(1x2) + Simple Batch Norm - Output size: 100 x 2 x 512
        with tf.name_scope('Conv_Pool_BN_6'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 256, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')


        # Seventh Layer: Conv (3x3) + Pool (1x2) - Output size: 100 x 1 x 512
        with tf.name_scope('Conv_Pool_7'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 512, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool(learelu, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

            self.cnnOut4d = pool

    def setupRNN(self):
        """ Create RNN layers and return output of these layers """
        # Collapse layer to remove dimension 100 x 1 x 512 --> 100 x 512 on axis=2
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # 2 layers of LSTM cell used to build RNN
        numHidden = 512
        cells = [tf.contrib.rnn.LSTMCell(
            num_units=numHidden, state_is_tuple=True, name='basic_lstm_cell') for _ in range(2)]
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        # Bi-directional RNN
        # BxTxF -> BxTx2H
        ((forward, backward), _) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        # Project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal(
            [1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        """ Create CTC loss and decoder and return them """
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])

        # Ground truth text as sparse tensor
        with tf.name_scope('CTC_Loss'):
            self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[
                                           None, 2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
            # Calculate loss for batch
            self.seqLen = tf.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                               ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))
        with tf.name_scope('CTC_Decoder'):
            # Decoder: Best path decoding or Word beam search decoding
            if self.decoderType == DecoderType.BestPath:
                self.decoder = tf.nn.ctc_greedy_decoder(
                    inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.BeamSearch:
                self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=True)
            elif self.decoderType == DecoderType.WordBeamSearch:
                # Import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
                word_beam_search_module = tf.load_op_library(
                    './TFWordBeamSearch.so')

                # Prepare: dictionary, characters in dataset, characters forming words
                chars = codecs.open(FilePaths.wordCharList.txt, 'r').read()
                wordChars = codecs.open(
                    FilePaths.fnWordCharList, 'r').read()
                corpus = codecs.open(FilePaths.corpus.txt, 'r').read()

                # # Decoder using the "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W) mode of word beam search
                # decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 25, 'NGramsForecastAndSample', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

                # Decoder using the "Words": only use dictionary, no scoring: O(1) mode of word beam search
                self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(
                    self.ctcIn3dTBC, dim=2), 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

        # Return a CTC operation to compute the loss and CTC operation to decode the RNN output
        return self.loss, self.decoder

    def setupTF(self):
        """ Initialize TensorFlow """
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.Session()  # Tensorflow session
        saver = tf.train.Saver(max_to_keep=3)  # Saver saves model to file
        modelDir = '../model/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # Is there a saved model?
        # If model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)
        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSpare(self, texts):
        """ Convert ground truth texts into sparse tensor for ctc_loss """
        indices = []
        values = []
        shape = [len(texts), 0]  # Last entry must be max(labelList[i])
        # Go over all texts
        for (batchElement, texts) in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            print(texts)
            labelStr = []
            for c in texts:
                 print(c, '|', end='')
                 labelStr.append(self.charList.index(c))
            print(' ')
            labelStr = [self.charList.index(c) for c in texts]
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput):
        """ Extract texts from output of CTC decoder """
        # Contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(Model.batchSize)]
        # Word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(Model.batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        # TF decoders: label strings are contained in sparse tensor
        else:
            # Ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]
            # Go over all indices and save mapping: batch -> values
            idxDict = {b : [] for b in range(Model.batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)
        # Map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch, batchNum):
        """ Feed a batch into the NN to train it """
        sparse = self.toSpare(batch.gtTexts)
        rate = 0.001 # if you use the pretrained model to continue train
        #rate = 0.01 if self.batchesTrained < 10 else (
        #    0.001 if self.batchesTrained < 2750 else 0.001) # variable learning_rate is used from trained from scratch
        evalList = [self.merge, self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse, self.seqLen: [Model.maxTextLen] * Model.batchSize, self.learningRate: rate}
        (loss_summary, _, lossVal) = self.sess.run(evalList, feedDict)
        # Tensorboard: Add loss_summary to writer
        self.writer.add_summary(loss_summary, batchNum)
        self.batchesTrained += 1
        return lossVal

    def return_rnn_out(self, batch, write_on_csv=False):
        """Only return rnn_out prediction value without decoded"""
        numBatchElements = len(batch.imgs)
        decoded, rnnOutput = self.sess.run([self.decoder, self.ctcIn3dTBC],
                                {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements})

        decoded = rnnOutput
        print(decoded.shape)

        if write_on_csv:
            s = rnnOutput.shape
            b = 0
            csv = ''
            for t in range(s[0]):
                for c in range(s[2]):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            open('mat_0.csv', 'w').write(csv)

        return decoded[:,0,:].reshape(100,80)

    def inferBatch(self, batch):
        """ Feed a batch into the NN to recognize texts """
        numBatchElements = len(batch.imgs)
        feedDict = {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        # # Dump RNN output to .csv file
        # decoded, rnnOutput = self.sess.run([self.decoder, self.rnnOutput], {
        #                                    self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * Model.batchSize})
        # s = rnnOutput.shape
        # b = 0
        # csv = ''
        # for t in range(s[0]):
        #     for c in range(s[2]):
        #         csv += str(rnnOutput[t, b, c]) + ';'
        #     csv += '\n'
        # open('mat_0.csv', 'w').write(csv)

        texts = self.decoderOutputToText(decoded)
        return texts

    def save(self):
        """ Save model to file """
        self.snapID += 1
        self.saver.save(self.sess, '../model/snapshot',
                        global_step=self.snapID)
