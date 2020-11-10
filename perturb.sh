#!/bin/bash
for data in   "cora" "citeseer" "polblogs"  
do
	mkdir -p perturbed/$data
	for i in 1 2 3 4 5 6 7 8 9 10 
	do
		for j in 0.5 #0.3 0.4 0.5 #0.05 0.1 0.15 0.2  0.25
		do
		    CUDA_VISIBLE_DEVICES=1 python  generate_attack.py --dataset $data --ptb_rate $j --seed $i 
		done
	done
done
