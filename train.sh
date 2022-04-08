#!/bin/bash

python main.py --exp_name=test --dataset=scanobjectnn --model=point-tnt --num_points=1024 --batch_size=32 --lr=0.001 --epochs=500 --beta=1.0
