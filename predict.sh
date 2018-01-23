#!/bin/bash

set -e 
set -u
baseDir="/data00/home/chenxiaoshuai/ml/segmentor/"

bin="${baseDir}/seg_keras_dnn.py"
mode="predict"
model="${baseDir}/model/dnn.model.less.hdf5"
vecFile="${baseDir}/renmin.vector.txt"
predictFile="${baseDir}/data/renmin.zi.flag.shuf.80w.txt"
predictFile="${baseDir}/data/x.txt"

if [ $# -ge 1 ];then
  predictFile="$1"
  echo ${predictFile}
fi

python ${bin} --mode=${mode} --model="${model}" --vector=${vecFile} --predict=${predictFile} #--flag
