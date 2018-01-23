#!/usr/bin/python
#coding:utf-8

#=======================================================================================
#        Copyright : hao666.info
#        File : seg_keras_dnn.py
#        Author : chenxs
#        Modify : 20180122 20:35:11
#        Description : 
#=======================================================================================
#基于DNN的序列标注分词
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import save_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import argparse

def parseArgs (progName) :
    progDesc = "program description : %s " % (progName)
    parse = argparse.ArgumentParser(progDesc)
    #parse.add_argument("vec", help = "vector path")
    parse.add_argument("-v", "--vector", type = str, help ="vector file")
    parse.add_argument("-m", "--mode", type = str, help ="train or test")
    parse.add_argument("-t", "--train", type = str, help ="train file")
    parse.add_argument("-M", "--model", type = str, help ="model file for predict")
    parse.add_argument("-p", "--predict", type = str, help ="predict file") #default = 1, choices = [1, 2, 4]
    parse.add_argument("-f", "--flag", help ="predict file with class flag or not", action="store_true")
    args = parse.parse_args()
    #print args
    return args

class SegmentorDNN (object) :
    ctxCnt_ = 5 # input = 2 * ctxCnt + 1
    windowLen_ = 2 * ctxCnt_ + 1
    
    vocabSize_ = 1000
    embedSize_ = 100
    
    padding_ = "</s>"
    
    xLine_ = []
    yLine_ = []

    X_ = []
    Y_ = []

    xTrain_ = []
    yTrain_ = []

    xTest_ = []
    yTest_ = []

    vocab_ = {}
    w2v_ = {}
    w2vNP_ = np.array((1, 1));
    
    flag_  = {'B' : 0, 'E' : 1, 'M' : 2, 'S' : 3}
    flagR_ = {0 : 'B', 1 : 'E', 2 : 'M', 3 : 'S'}

    def __init__ (self) :
        pass

    def train(self):
        print "[info]train size = ", len(self.xTrain_)
        #print "[info]yTrain_ = ", self.yTrain_
        model = Sequential()
        lEmbd = Embedding(self.vocabSize_, self.embedSize_, weights=[self.w2vNP_], input_length= self.windowLen_, trainable=False)
        model.add(lEmbd)
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.flag_), activation='softmax'))
        
        filepath = "checkpoint/seg_dnn_{epoch:02d}_{val_acc:.2f}.hdf5"
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only=True, mode='max')
        callbacksList = [checkpoint]
        model.fit(self.xTrain_, self.yTrain_, batch_size = 128, epochs = 50, verbose = 1, callbacks = callbacksList, validation_split = 0.1)

        save_model(model, "model/dnn.model.hdf5");
        loss, accuracy = model.evaluate(self.xTest_, self.yTest_, verbose=1)
        print('[info]loss: %f' % (loss))
        print('[info]Accuracy: %f' % (accuracy * 100))

    def predict(self, modelFile, testFile, withFlag = False):
        model = load_model(modelFile) 
        with open(testFile) as fd:
            line = fd.readline()
            lineCnt = 0
            while line != "" :
                lineCnt += 1;
                if lineCnt % 200 == 0:
                    sys.stderr.write("[info]line = %d\n" % lineCnt)
                line = line.strip().decode("utf8")
                line = line[0 : : 2] if withFlag else line
                tmpXList = list(line)
                #句尾padding，这样才能训练句首/句尾字
                tmpXList = [self.padding_] * (self.ctxCnt_) + tmpXList + [self.padding_] * (self.ctxCnt_)
                
                tmpCnt = len(tmpXList)
                #curX = np.zeros((tmpCnt - self.windowLen_ + 1, self.windowLen_))

                curX = np.array([0] * (tmpCnt - self.windowLen_ + 1) * (self.windowLen_)).reshape((tmpCnt - self.windowLen_ + 1, self.windowLen_))
                for ti in range(tmpCnt - self.windowLen_ + 1) :
                    for tj in range(self.windowLen_):
                        tmpZi = tmpXList[ti + tj]
                        curX[ti][tj] =  self.vocab_[tmpZi] if tmpZi in self.vocab_ else 0;
                #print "[info]will predict using model"
                curPreClass = model.predict_classes(curX, batch_size = len(curX))
                curPreClass = map( lambda x : self.flagR_[x], curPreClass)
                #print line
                #print curPreClass
                for oi in range(len(curPreClass)) :
                    if oi > 0 and ( curPreClass[oi] == 'B' or curPreClass[oi] == 'S'):
                        sys.stdout.write("%s" % (" "))
                    sys.stdout.write("%s" % (line[oi].encode("utf8")))
                sys.stdout.write("%s" % ("\n"))
                    
                line = fd.readline()
        

    def loadWord2Vec(self, inFile) :
        sys.stderr.write("[info]begin loadWord2Vec\n")
        with open(inFile) as fd :
            line = fd.readline()
            line = line.strip().split(" ");
            self.vocabSize_ = int(line[0])
            self.embedSize_ = int(line[1])
            self.w2vNP_ = np.zeros((self.vocabSize_, self.embedSize_))
            #print "vocab = ", self.vocabSize_ , ", embedSize = ",  self.embedSize_

            lineCnt = 0
            line = fd.readline()
            while line != "" :
                #print line, len(line)
                line = line.strip().split(" ")
                self.w2v_[lineCnt] = np.array(line[1 : ], dtype = 'float32')
                self.vocab_[line[0].decode("utf8")] = lineCnt
                self.w2vNP_[lineCnt] = np.array(line[1 : ], dtype = 'float32')
                lineCnt += 1;
                line = fd.readline()
        sys.stderr.write("[info]end loadWord2Vec\n")

    def loadTrain(self, inFile):
        print "[info]begin loading data"
        #inFile
        #现在看到的都是一下把训练集读进来，不能一条一条的读入进行训练吗，或者是一个batch一个batch的读入,可以使用train_on_batch，自己写batch的itertor
        #读取输入，按windowLen划分,每个输入句子制作为 max(l -  windowLen  + 1, 1)个训练集
        #预测中间词

        with open(inFile) as fd :
            line = fd.readline()
    
            lineCnt = 0
            while line != "" :
                lineCnt += 1;
                if lineCnt % 20000 == 0 :
                    print "[info]line = ", lineCnt
                #print line, len(line)
                line = line.strip().decode("utf8")
                #self.xLine_.append(list(line[0 : : 2]))
                #self.yLine_.append(list(line[1 : : 2]))
                tmpXList = list(line[0 : : 2])
                tmpYList = list(line[1 : : 2])
                
                #句尾padding，这样才能训练句首/句尾字
                tmpXList = [self.padding_] * (self.ctxCnt_) + tmpXList + [self.padding_] * (self.ctxCnt_)
                tmpYList = ['S'] * (self.ctxCnt_) + tmpYList + ['S'] * (self.ctxCnt_)
                tmpCnt = len(tmpXList)

                if tmpCnt < self.windowLen_ :
                    pass
                else :
                    for ti in range(tmpCnt - self.windowLen_ + 1) :
                        self.X_.append(tmpXList[ti: ti + self.windowLen_])
                        self.Y_.append(tmpYList[ti: ti + self.windowLen_])
                line = fd.readline()
                #if lineCnt >= 300 :
                #    break;
        #self.dump()
        self.toVocabId()
        #self.dump()
        #print self.xLine_
        #print self.xLine_
        print "[info]end loading data"

    def splitTrainTest(self, perc) : 
        print "[info]begin splitTrainTest"
        trainCnt = int((1 - perc) * len(self.X_))
        self.xTrain_ = np.asarray(self.X_[: trainCnt])
        #self.yTrain_ = np.asarray(self.Y_[: trainCnt])
        self.yTrain_ = np.zeros((trainCnt, len(self.flag_)))
        for i in range(trainCnt) :
            self.yTrain_[i][self.Y_[i][self.ctxCnt_]] = 1 #self.Y_[i][self.ctxCnt_]中间标签的标签id
            #print self.yTrain_[i]
            #t = np.zeros((len(self.flag_),))
            #self.yTrain_[i] = t

        self.xTest_ = np.asarray(self.X_[trainCnt :])
        #self.yTest_ = np.asarray(self.Y_[trainCnt :])
        #self.yTest_ = np.asarray(self.Y_)[trainCnt :, self.ctxCnt_ : self.ctxCnt_ + 1]
        #for i in range(len(self.yTest_)) :
        #    t = np.zeros((len(self.flag_),))
        #    t[self.yTest_[i][0]] = 1
        #    #self.yTest_[i] = t
        #    print self.yTest_[i]
        self.yTest_ = np.zeros((len(self.X_) - trainCnt, len(self.flag_)))
        for i in range(trainCnt, len(self.X_)) :
            self.yTest_[i - trainCnt ][self.Y_[i][self.ctxCnt_]] = 1
            #print self.yTrain_[i - trainCnt]
        print "[info]end splitTrainTest"
    
    def toVocabId(self):
        print "[info]begin toVeocbId"
        for i in range(len(self.X_)) :
            self.X_[i] = map(lambda x : self.vocab_[x] if x in self.vocab_ else 0, self.X_[i])
            self.Y_[i] = map(lambda y : self.flag_[y] if y in self.flag_ else 0, self.Y_[i])
            #print self.Y_[i]
        print "[info]end toVeocbId"


    def dump(self):
        for i in range(len(self.X_)) :
            for j in range(len(self.X_[i])) :
                if isinstance(self.Y_[i][j], unicode):
                    sys.stderr.write("%s%s " % (self.X_[i][j].encode("utf8"), self.Y_[i][j].encode("utf8") ) )
                else:
                    print self.Y_[i][j]
                    print self.Y_[i]
                    sys.stderr.write("%d_%d " % (self.X_[i][j], self.Y_[i][j]))
            sys.stderr.write("\n")

if __name__ == "__main__" :
    
    args = parseArgs (sys.argv[0])
    mode = args.mode
    seg = SegmentorDNN ()
    seg.loadWord2Vec(args.vector)

    if mode == "train" :
        seg.loadTrain(args.train)
        seg.splitTrainTest(0.2) 
        seg.train()
    elif mode == "predict":
        seg.predict(args.model, args.predict, args.flag)
    
