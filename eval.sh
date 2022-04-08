#!/bin/bash

# evaluate on scanobjectnn
python main.py --eval --model_path=pretrained/scanobjectnn/model_final.t7 --dataset=scanobjectnn --model=point-tnt --num_points=1024 --test_batch_size=8 --num_data_workers=2

# evaluate on modelnet40
python main.py --eval --model_path=pretrained/modelnet40/model_final.t7 --dataset=modelnet40 --model=point-tnt --num_points=1024 --test_batch_size=8 --num_data_workers=2
