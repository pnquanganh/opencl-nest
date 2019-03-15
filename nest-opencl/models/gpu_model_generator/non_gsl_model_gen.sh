#!/bin/bash

# MODEL_NAME=$1
PARSING_BINARY_DIR=/home/pham0071/cfe-6.0.1.src/build/bin
PARSING_BINARY=model_parsing
NEST_DIR=/home/pham0071/nest-project/nest-gpu/nest-with-gpu-style-gsl


for model_name in "$@"
do
    echo $model_name
    $PARSING_BINARY_DIR/$PARSING_BINARY -p=$NEST_DIR $NEST_DIR/models/$model_name.cpp | python ./non_gsl_gpu_model_gen.py $model_name
done
