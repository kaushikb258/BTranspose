#!/bin/bash

IMAGE_PATH=hpcregistry.hpc.ford.com/kbalak18/repo3:1
OUTPUT_PATH=/s/kbalak18/Facebook-Dino-master/pose_kb1/logs/

#Berfore Changes
runpytorch-dist -NGPUS 16 -x facebook-dino.sh -i ${IMAGE_PATH} -np 16 -l ${OUTPUT_PATH} 
