#!/bin/bash

start=401
label=0
label_size=751

while (($start < $label_size))
do
    end=$((start + 349))
    if (($end >= $label_size)); then
        end=$((label_size - 1))
    fi

    job_name="dock_${label}_${start}_${end}"
    echo $job_name
    qsub -N ${job_name} -v LABEL=${label} START=${start} END=${end} dock.pbs
    echo "------------------"
    start=$((end + 1))
done

#class 0 size: 7625.
#class 1 size: 1158.
#class 2 size: 3001.
#class 3 size: 1054.
#class 4 size: 968.
#class 5 size: 1890.
#class 6 size: 1663.
#class 7 size: 602.
#class 8 size: 573.
#class 9 size: 566.
#class 10 size: 897.
#class 11 size: 417.
#class 12 size: 374.
#class 13 size: 337.