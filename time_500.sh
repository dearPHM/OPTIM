# rm -rf results
mkdir results

BRAIN='src/brain_main.py --model=cnn --dataset=cifar --epochs=500 --num_users=21 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.15 --verbose=0 --local_bs=500'
OPTIM="${BRAIN} --optim=True"

QUEUE='--maxqueue=50'

NONIID='--iid=0'

for i in $(seq 1 10); do
    echo "loop $i start"
    
    python $BRAIN
    python $OPTIM

    python $BRAIN $NONIID
    python $OPTIM $NONIID

    python $BRAIN $QUEUE
    python $OPTIM $QUEUE

done
