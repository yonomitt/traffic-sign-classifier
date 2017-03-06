#!/bin/bash

python3 project2.py -l 0.0001 -n Inception_dropout_fc -p CLAHE -e 200 -k 0.5 -t ../data/traffic-signs-data/train_pp.p
python3 project2.py -l 0.0001 -n Inception2_dropout_fc -p CLAHE -e 200 -k 0.5 -t ../data/traffic-signs-data/train_pp.p


