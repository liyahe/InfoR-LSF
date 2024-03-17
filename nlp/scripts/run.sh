task=$1
debug_mode=${2-false}
epochs=15
max_seq_length=128
if [ $task == "imdb" ] || [ $task == "yelp" ]; then
    max_seq_length=256
fi
log_file=logs/$(date "+%Y-%m-%d")-$task.log

for num_samples in 50 100 200 500 1000; do
    for data_seed in 13 21 42 87 100; do

        echo -e "\e[34m task: $task, num_samples: $num_samples, data_seed: $data_seed, max_seq_length: $max_seq_length, log_file: $log_file debug_mode: $debug_mode \e[0m"

        # baseline
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/baseline/trainSize$num_samples/seed$data_seed \
            --task_name $task --model_type bert \
            --do_eval --max_seq_length $max_seq_length --num_train_epochs $epochs --overwrite_output_dir \
            --outputfile results/baseline_withdev_results.csv --do_lower_case \
            --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
            --eval_types train dev --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --log_file $log_file --disable_wandb \
            --debug $debug_mode

        # # vibert
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/vibert/trainSize$num_samples/seed$data_seed --task_name $task --model_type bert --do_eval --max_seq_length $max_seq_length --num_train_epochs 25 --overwrite_output_dir \
            --outputfile results/vibbert_withdev_results.csv --do_lower_case --ib_dim 384 \
            --beta 1e-05 --ib --learning_rate 2e-5 --do_train --sample_train --num_samples $num_samples --eval_types dev train --kl_annealing linear --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb

        # vibert-ablation beta=0
        # python run_glue.py  --model_name_or_path  bert-base-uncased  \
        #     --output_dir output/low_resource/$task/vibert_beta0/trainSize$num_samples/seed$data_seed  --task_name $task --model_type bert  --do_eval\
        #     --max_seq_length $max_seq_length  --num_train_epochs 25 --overwrite_output_dir \
        #     --outputfile results/vibbert_ablation_withdev_results.csv  --do_lower_case  --ib_dim 384 \
        #     --beta 0.0 --ib --learning_rate 2e-5  --do_train --sample_train --num_samples $num_samples\
        #     --eval_types dev train   --kl_annealing linear --seed 42 --data_seed $data_seed\
        #     --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb

        # vibert-deterministic
        # python run_glue.py  --model_name_or_path bert-base-uncased --seed 42\
        #       --output_dir output/tmp --task_name $task  --model_type bert\
        #       --do_eval --max_seq_length $max_seq_length --num_train_epochs 25  \
        #       --overwrite_output_dir --outputfile results/vibbert_ablation_withdev_results.csv \
        #       --do_lower_case  --ib_dim 384 --deterministic --learning_rate 2e-5\
        #       --do_train --sample_train --num_samples $num_samples --eval_types dev train\
        #       --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode

        # ifm
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/ifm/trainSize$num_samples/seed$data_seed --task_name $task --model_type bert \
            --do_eval --max_seq_length $max_seq_length --num_train_epochs $epochs --overwrite_output_dir \
            --outputfile results/ifm_withdev_results.csv --do_lower_case \
            --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
            --eval_types train dev --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb \
            --local_params '{"ifm":true,"ifm_epsilon":0.05,"ifm_alpha":1.0}'

        x_perc_or_num=0.05
        if [ $task == "snli" ]; then
            x_perc_or_num=1
        fi
        ir_beta=0.001
        if [ $task == "sts-b" ]; then
            ir_beta=0.01
        fi
        echo "x_perc_or_num: $x_perc_or_num, ir_beta: $ir_beta"

        # InfoR-LSF
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/irlsf/trainSize$num_samples/seed$data_seed --task_name $task --model_type bert \
            --do_eval --max_seq_length $max_seq_length --num_train_epochs 25 --overwrite_output_dir \
            --outputfile results/inforetention_wozaugsup_withdev_results.csv --do_lower_case \
            --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
            --ib_dim 384 --beta 1e-05 --ib --kl_annealing linear --eval_types train dev --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb \
            --local_params '{"sup_loss_on_z_":false,"InfoRetention":true,"ir_beta":'$ir_beta',"multi_stages":true,"ifm_epsilon":0.1,"z_perc_or_num":1.0,"mi_type":"lower_bound","x_perc_or_num":'$x_perc_or_num',"mask_type":"token_mask","only_mask_correct":false}'

        # InfoR-LSF ablation - ir_beta(alpha in paper)=0
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/irlsf_alpha0/trainSize$num_samples/seed$data_seed --task_name $task --model_type bert \
            --do_eval --max_seq_length $max_seq_length --num_train_epochs 25 --overwrite_output_dir \
            --outputfile results/inforetention_wozaugsup_woir_withdev_results.csv --do_lower_case \
            --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
            --ib_dim 384 --beta 1e-05 --ib --kl_annealing linear --eval_types train dev --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb \
            --local_params '{"sup_loss_on_z_":false,"InfoRetention":true,"ir_beta":0.0,"multi_stages":true,"ifm_epsilon":0.1,"z_perc_or_num":1.0,"mi_type":"lower_bound","x_perc_or_num":'$x_perc_or_num',"mask_type":"token_mask","only_mask_correct":false}'

        # InfoR-LSF ablation - ib_beta(beta in paper)=0
        python run_glue.py --model_name_or_path bert-base-uncased \
            --output_dir output/low_resource/$task/irlsf_ib_beta0/trainSize$num_samples/seed$data_seed --task_name $task --model_type bert \
            --do_eval --max_seq_length $max_seq_length --num_train_epochs 25 --overwrite_output_dir \
            --outputfile results/inforetention_wozaugsup_woib_withdev_results.csv --do_lower_case \
            --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
            --ib_dim 384 --beta 0.0 --ib --kl_annealing linear --eval_types train dev --seed 42 --data_seed $data_seed --evaluate_after_each_epoch --debug $debug_mode --log_file $log_file --disable_wandb \
            --local_params '{"sup_loss_on_z_":false,"InfoRetention":true,"ir_beta":'$ir_beta',"multi_stages":true,"ifm_epsilon":0.1,"z_perc_or_num":1.0,"mi_type":"lower_bound","x_perc_or_num":'$x_perc_or_num',"mask_type":"token_mask","only_mask_correct":false}'

        # inbatch_irlsf
        # python run_glue.py  --model_name_or_path bert-base-uncased \
        #        --output_dir output/low_resource/$task/inbatch_irlsf/trainSize$num_samples/seed$data_seed \
        #        --task_name $task  --model_type bert \
        #        --do_eval --max_seq_length $max_seq_length  --num_train_epochs 25 --overwrite_output_dir \
        #        --outputfile results/inbatch_irlsf_withdev_results.csv  --do_lower_case \
        #        --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples $num_samples \
        #        --ib_dim 384 --beta 1e-5 --ib --kl_annealing linear\
        #        --eval_types train dev --seed 42 --data_seed $data_seed\
        #        --evaluate_after_each_epoch --debug $debug_mode\
        #        --log_file $log_file --disable_wandb \
        #        --local_params '{"sup_loss_on_z_":false,"inbatch_irlsf":true,"ir_beta":$ir_beta,"ifm_epsilon":0.1,"z_perc_or_num":1.0,"mi_type":"lower_bound","x_perc_or_num":'$x_perc_or_num',"mask_type":"token_mask","only_mask_correct":false}'

    done
done
