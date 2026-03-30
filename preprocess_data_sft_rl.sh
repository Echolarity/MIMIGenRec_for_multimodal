set -e

export DATA_DIR="newdata/Amazon"

export CATEGORY="Industrial_and_Scientific"
# export CATEGORY="Office_Products"
# export CATEGORY="Toys_and_Games"

# output directory (sft/, rl/, new_tokens.json); default data/<category>
export OUTPUT_DIR="newdata/${CATEGORY}"

export TASK4_SAMPLE=-1 # sample all if -1
export SEED=42

python preprocess_data_sft_rl.py \
    --data_dir $DATA_DIR \
    --category $CATEGORY \
    --output_dir $OUTPUT_DIR \
    --seq_sample $TASK4_SAMPLE \
    --seed $SEED \
    --data_source $CATEGORY
