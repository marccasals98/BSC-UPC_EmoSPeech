#!/bin/bash
#SBATCH -o /home/usuaris/veussd/marc.casals/EmoSPeech2024/logs/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veussd/marc.casals/EmoSPeech2024/logs/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_8_classes_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marc.casals@bsc.es

python src/train.py \
	--model_output_folder "/home/usuaris/veussd/marc.casals/EmoSPeech2024/models" \
	--training_random_crop_secs 9 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 9 \
	--training_augmentation_prob 0.0 \
	--evaluation_augmentation_prob 0 \
	--feature_extractor 'WavLMExtractor' \
	--wavlm_flavor 'WAV2VEC2_XLSR_300M' \
	--feature_extractor_output_vectors_dimension 1024 \
	--text_feature_extractor 'NoneTextExtractor' \
	--front_end 'NoneFrontEnd' \
	--adapter 'NoneAdapter' \
	--seq_to_seq_method 'NoneSeqToSeq' \
	--seq_to_seq_input_dropout 0.0 \
	--seq_to_one_method 'AttentionPooling' \
	--seq_to_one_input_dropout 0.0 \
	--max_epochs 200 \
	--training_batch_size 16 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 1250 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 6 \
	--weighted_loss \
	--optimizer 'adamw' \
	--learning_rate 0.00005 \
	--learning_rate_multiplier 0.9 \
	--weight_decay 0.01 \
	--use_weights_and_biases