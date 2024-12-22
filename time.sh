# rm -rf results
mkdir results

BRAIN='src/brain_main.py --model=cnn --dataset=cifar --epochs=200 --num_users=21 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.15 --verbose=0 --local_bs=500'
OPTIM="${BRAIN} --optim=True"

BYZ='--byzantines=10'
BYZ_SCORE="$BYZ --score_byzantines=5"

HISTORY_ORIGINAL='--history=0'
HISTORY_STREAK='--history=1'
HISTORY_AVERAGE='--history=2'

QUEUE='--maxqueue=50'

NONIID='--iid=0'

# ADV='--advanced_threshold=True'

for i in $(seq 1 7); do
    echo "loop $i start"
    
    # python $BRAIN $BYZ
    # python $BRAIN $BYZ $NONIID

    # python $OPTIM $BYZ $HISTORY_ORIGINAL
    # python $OPTIM $BYZ $HISTORY_STREAK
    
    # python $OPTIM $BYZ $NONIID $HISTORY_ORIGINAL
    # python $OPTIM $BYZ $NONIID $HISTORY_STREAK

    # python $BRAIN $BYZ $NONIID $QUEUE

    python $OPTIM $HISTORY_AVERAGE $BYZ
    python $OPTIM $HISTORY_AVERAGE $NONIID $BYZ

    # python $QUEUE $NONIID
    # python $OPTIM $HISTORY_ORIGINAL $QUEUE $NONIID

    # python $OPTIM $BYZ $HISTORY_ORIGINAL $QUEUE
    # python $OPTIM $BYZ $HISTORY_STREAK $QUEUE
    
    # python $OPTIM $BYZ $NONIID $HISTORY_ORIGINAL $QUEUE
    # python $OPTIM $BYZ $NONIID $HISTORY_STREAK $QUEUE

done
