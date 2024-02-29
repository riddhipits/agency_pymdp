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

average_self_rating_pos=$(echo $folders | xargs cat | grep "endofexp_self_rating_pos" |  ./average.sh)
average_self_rating_neg=$(echo $folders | xargs cat | grep "endofexp_self_rating_neg" |  ./average.sh)
average_self_rating_zero=$(echo $folders | xargs cat | grep "endofexp_self_rating_zero" |  ./average.sh)
average_other_rating_pos=$(echo $folders | xargs cat | grep "endofexp_other_rating_pos" |  ./average.sh)
average_other_rating_neg=$(echo $folders | xargs cat | grep "endofexp_other_rating_neg" |  ./average.sh)
average_other_rating_zero=$(echo $folders | xargs cat | grep "endofexp_other_rating_zero" |  ./average.sh)
average_p_self_action_press=$(echo $folders | xargs cat | grep "endofexp_p_self_action_press" |  ./average.sh)

variance_self_rating_pos=$(echo $folders | xargs cat | grep "endofexp_self_rating_pos" | ./variance.sh $average_self_rating_pos)
variance_self_rating_neg=$(echo $folders | xargs cat | grep "endofexp_self_rating_neg" | ./variance.sh $average_self_rating_neg)
variance_self_rating_zero=$(echo $folders | xargs cat | grep "endofexp_self_rating_zero" | ./variance.sh $average_self_rating_zero)
variance_other_rating_pos=$(echo $folders | xargs cat | grep "endofexp_other_rating_pos" | ./variance.sh $average_other_rating_pos)
variance_other_rating_neg=$(echo $folders | xargs cat | grep "endofexp_other_rating_neg" | ./variance.sh $average_other_rating_neg)
variance_other_rating_zero=$(echo $folders | xargs cat | grep "endofexp_other_rating_zero" | ./variance.sh $average_other_rating_zero)
variance_p_self_action_press=$(echo $folders | xargs cat | grep "endofexp_p_self_action_press" | ./variance.sh $average_p_self_action_press)

overview_file="result.txt"
if [ ! -e $overview_file ]
then
    echo "Log of experiments so far." >> $overview_file
fi
echo "Experiment result:"
description=$(awk '/description:/{flag=1;next}/^---$/{flag=0}flag' config.yaml | head -n 5)
echo "experiment: $description"  | tee -a $overview_file
echo "runs: $repeats" | tee -a $overview_file
echo " "
echo " SELF RATING "
echo "average self rating pos: $average_self_rating_pos" | tee -a $overview_file
echo "average self rating neg: $average_self_rating_neg" | tee -a $overview_file
echo "average self rating zero: $average_self_rating_zero" | tee -a $overview_file
echo " "
echo "variance self rating pos: $variance_self_rating_pos" | tee -a $overview_file
echo "variance self rating neg: $variance_self_rating_neg" | tee -a $overview_file
echo "variance self rating zero: $variance_self_rating_zero" | tee -a $overview_file
echo " "
echo " "
echo " OTHER RATING "
echo "average other rating pos: $average_other_rating_pos" | tee -a $overview_file
echo "average other rating neg: $average_other_rating_neg" | tee -a $overview_file
echo "average other rating zero: $average_other_rating_zero" | tee -a $overview_file
echo " "
echo "variance other rating pos: $variance_other_rating_pos" | tee -a $overview_file
echo "variance other rating neg: $variance_other_rating_neg" | tee -a $overview_file
echo "variance other rating zero: $variance_other_rating_zero" | tee -a $overview_file
echo " "
echo " "
echo " PROB(SELF ACTION) "
echo "average prob(self action): $average_p_self_action_press" | tee -a $overview_file
echo "variance prob(self action): $variance_p_self_action_press" | tee -a $overview_file
echo "sample folders: $(echo $folders | sed -e 's/\n/ /g')" >> $overview_file
echo "================================================================================" >> $overview_file
echo "Done parsing results."