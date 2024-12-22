mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Performance
        python src/baseline_main.py --epochs=200 --lr=9.0 --verbose=0
        python src/federated_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0

        # Byzantine
        ## FedAvg
        python src/federated_main.py --iid=${IID} --epochs=200 --byzantines=5 --frac=0.1 --verbose=0
        python src/federated_main.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --verbose=0
        python src/federated_main.py --iid=${IID} --epochs=200 --byzantines=11 --frac=0.1 --verbose=0
        python src/federated_main.py --iid=${IID} --epochs=200 --byzantines=15 --frac=0.1 --verbose=0
        ## FedAsync
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=5 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=11 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=15 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        ## BRAIN
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=11 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=15 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0

        # Threshold
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.3 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.4 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.5 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.6 --verbose=0

        # Score Byzantine
        # ## Byzantine 0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        ## Byzantine 5
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0

        # Staleness
        ## FedAsync
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=8 --alpha=0.6 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=16 --alpha=0.6 --verbose=0
        python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=32 --alpha=0.6 --verbose=0
        # python src/fedAsync_main.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=64 --alpha=0.6 --verbose=0
        ## BRAIN
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=8 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=32 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        # python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=64 --diff=0.55 --window=4 --threshold=0.0 --verbose=0

        # Quorum
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.25 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/brain_main.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.75 --window=4 --threshold=0.2 --verbose=0
    done
done
