task_name=$1

for task in $task_name; do
    max_seq_length=128
    if [ $task == "imdb" ] || [ $task == "yelp" ]; then
        max_seq_length=256
    fi
    for num_samples in 50 100 200 500 1000; do
        echo $task $num_samples
        if [ $task == "sts-b" ]; then
            for file in baseline_withdev_results.csv ifm_withdev_results.csv vibbert_withdev_results.csv inforetention_wozaugsup_withdev_results.csv inforetention_wozaugsup_woir_withdev_results.csv inforetention_wozaugsup_woib_withdev_results.csv; do
                python tools/gather_result.py --file $file --condition '{"task_name":"'$task'","num_samples":'$num_samples',"deterministic":false,"max_seq_length":'$max_seq_length'}' \
                    --key pearson_dev --test_key pearson_test --test_key2 pearson_train
            done
        else
            for file in baseline_withdev_results.csv ifm_withdev_results.csv vibbert_withdev_results.csv inforetention_wozaugsup_withdev_results.csv inforetention_wozaugsup_woir_withdev_results.csv inforetention_wozaugsup_woib_withdev_results.csv; do
                python tools/gather_result.py --file $file --condition '{"task_name":"'$task'","num_samples":'$num_samples',"deterministic":false,"max_seq_length":'$max_seq_length'}'
            done
        fi
    done
done
