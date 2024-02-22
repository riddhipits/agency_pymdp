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
    echo "Starting experiment $i"
    sleep $i && python run_experiment.py $config_name experiments 1>/dev/null 2>>python.log &
done
echo "Spawned $repeats experiments waiting for results."
wait
echo "Done Running, parsing results."

# take the $repeats most recent folders
((topN=repeats))
folders=$(ls -t experiments | head -$topN | awk '{print "experiments/" $0 "/output.yaml"}')
echo "Checking for output files in experiments directory..."
ls -l experiments/
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

echo "Config name: $config_name"
echo "Repeats: $repeats"



average_self_rating=$(echo $folders | xargs cat | grep "endofexp_self_rating" |  ./average.sh)
average_other_rating=$(echo $folders | xargs cat | grep "endofexp_other_rating" |  ./average.sh)
average_p_self_action=$(echo $folders | xargs cat | grep "endofexp_p_self_action" |  ./average.sh)
variance_self_rating=$(echo $folders | xargs cat | grep "endofexp_self_rating" | ./variance.sh $average_self_rating)
variance_other_rating=$(echo $folders | xargs cat | grep "endofexp_other_rating" | ./variance.sh $average_other_rating)
variance_p_self_action=$(echo $folders | xargs cat | grep "endofexp_p_self_action" | ./variance.sh $average_p_self_action)

overview_file="result.txt"
if [ ! -e $overview_file ]
then
    echo "Log of experiments so far." >> $overview_file
fi
echo "Experiment result:"
description=$(awk '/description:/{flag=1;next}/^---$/{flag=0}flag' config.yaml | head -n 5)
echo "experiment: $description"  | tee -a $overview_file
echo "runs: $repeats" | tee -a $overview_file

echo "average self rating: $average_self_rating" | tee -a $overview_file
echo "average other rating: $average_other_rating" | tee -a $overview_file
echo "average prob(self action): $average_p_self_action" | tee -a $overview_file
echo "variance self rating: $variance_self_rating" | tee -a $overview_file
echo "variance other rating: $variance_other_rating" | tee -a $overview_file
echo "variance prob(self action): $variance_p_self_action" | tee -a $overview_file

echo "sample folders: $(echo $folders | sed -e 's/\n/ /g')" >> $overview_file
echo "================================================================================" >> $overview_file
echo "Done parsing results."
