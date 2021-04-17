#!/bin/bash

export BUCKET_NAME=torch_model_info
export JOB_NAME=train_vae
export JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models