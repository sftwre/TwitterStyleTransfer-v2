#!/bin/bash

BUCKET_NAME=torch_model_info
JOB_NAME=vae_job_$(date +%Y%m%d_%H%M%S)
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
  --region=us-central1 \
  --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-7 \
  --scale-tier=CUSTOM \
  --master-machine-type=n1-standard-8 \
  --master-accelerator=type=nvidia-tesla-t4,count=4 \
  --job-dir=${JOB_DIR} \
  --package-path=./trainer \
  --module-name=trainer.train_vae \
  -- \
  --epochs=2 \
  --log=True