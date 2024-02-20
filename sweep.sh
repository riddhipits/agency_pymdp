#!/usr/bin/env sh

config_name=$1
if [ -z $config_name ]
then
    echo "Run as $0 <config_file> <repeat_count>"
    exit
fi

repeats=$2
if [ -z $repeats ]
then
    echo "Run as $0 <config_file> <repeat_count>"
    exit
fi

rm python.log 2> /dev/null

echo "Starting $repeats runs of $config_name"
for i in $(seq $repeats)
do
    sleep $i && python run_experiment.py $config_name experiments 1>/dev/null 2>>python.log &
done
echo "Spawned $repeats experiments waiting for results."
wait
echo "Done Running, parsing results."

# take the $repeats most recent folders
((topN=repeats))
folders=$(ls -t experiments | head -$topN | awk '{print "experiments/" $0 "/output.yaml"}')
for i in $folders
do
    if [ ! -e $i ]
    then
        echo "Failed to find the expected output. Aborting!"
        echo "Nothing got added to the result log."
        echo "Check the python logs to see what went wrong."
        exit
    fi
done


average_success_exp=$(echo $folders | xargs cat | grep -c "found: true")
average_length_exp=$(echo $folders | xargs cat | grep "length" |  ./average.sh)
variance_length_exp=$(echo $folders | xargs cat | grep "length" | ./variance.sh $average_length_exp)
average_coverage_exp=$(echo $folders | xargs cat | grep "coverage" | ./average.sh)
variance_coverage_exp=$(echo $folders | xargs cat | grep "coverage" | ./variance.sh $average_coverage_exp)
average_coverage_auc=$(echo $folders | xargs cat | grep "auc_norm" |  ./average.sh)
variance_coverage_auc=$(echo $folders | xargs cat | grep "auc_norm" | ./variance.sh $average_coverage_auc)
overview_file="result.txt"
if [ ! -e $overview_file ]
then
    echo "Log of experiments so far." >> $overview_file
fi
echo "Experiment result:"
description=$(awk '/description:/{flag=1;next}/^---$/{flag=0}flag' config.yaml | head -n 5)
echo "experiment: $description"  | tee -a $overview_file
echo "runs: $repeats" | tee -a $overview_file
echo "success rate: $average_success_exp" | tee -a $overview_file
echo "average experiment length: $average_length_exp" | tee -a $overview_file
echo "stdv experiment length: $variance_length_exp" | tee -a $overview_file
echo "average coverage: $average_coverage_exp" | tee -a $overview_file
echo "stdv coverage: $variance_coverage_exp" | tee -a $overview_file
echo "average coverage auc: $average_coverage_auc" | tee -a $overview_file
echo "stdv coverage auc: $variance_coverage_auc" | tee -a $overview_file
echo "sample folders: $(echo $folders | sed -e 's/\n/ /g')" >> $overview_file
echo "================================================================================" >> $overview_file
echo "Done parsing results."
