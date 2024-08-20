#!/bin/bash

# Sync models from S3 to the local directory
aws s3 sync s3://tmunan-models/tensorrt /root/app/tensorrt

# Execute application
exec uvicorn tmunan.imagine_app.imagine:app --host 0.0.0.0 --port 8090
