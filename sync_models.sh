#!/bin/bash

MODEL=$1

cp ${MODEL} /mnt/data/storage/infer_models/
scp ${MODEL} 192.168.16.154:/mnt/data/storage/infer_models/
