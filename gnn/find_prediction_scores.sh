#!/bin/bash
for i in 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56
do
   seed=$((i-40))
   python find_predition_scores.py -run $i -seed $seed
   echo "-------------------------------------------"
done