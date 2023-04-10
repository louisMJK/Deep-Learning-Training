./train.sh 1 --model "vit_tiny" \
    --batch_size 128 \
    --opt adam --lr 0.001 \
    --sched cosine --epochs 2 
