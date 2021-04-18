#!/bin/bash

export BUCKET_NAME=torch_model_info
export JOB_NAME=pytorch_job_$(date +%Y%m%d_%H%M%S)
export JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models