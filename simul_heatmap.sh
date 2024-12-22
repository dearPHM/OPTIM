mkdir simul_heatmap

# SIMULATION
RES_PATH="./simul_heatmap"
REPEAT=100
# CHAIN=("ETHEREUM" "POLYGON")
CHAIN=("ETHEREUM")
QTO=7881  # 0 for infinity

TIMES=86400.0
AMOUNTS=1000

for C in ${CHAIN[@]}; do
    SIZE=$([ ${C} == "ETHEREUM" ] && echo 155 || echo 72)
    INTERVAL=$([ ${C} == "ETHEREUM" ] && echo 12.06 || echo 2.07)

    DIFF=(16 32 64 128)  # Higher == Easier
    for D in ${DIFF[@]}; do

        QUORUM=(5 10 15 20)
        for QC in ${QUORUM[@]}; do

            EPOCH=(1 2 4 8 16)
            for E in ${EPOCH[@]}; do
                echo python simulate/nodes.py --verbose 0 --repeat ${REPEAT} --epoch ${E} --qc ${QC} --d ${D} --size ${SIZE} --interval ${INTERVAL} --path "./save_heatmap" --qto ${QTO} --times ${TIMES} --amounts ${AMOUNTS}
                     python simulate/nodes.py --verbose 0 --repeat ${REPEAT} --epoch ${E} --qc ${QC} --d ${D} --size ${SIZE} --interval ${INTERVAL} --path "./save_heatmap" --qto ${QTO} --times ${TIMES} --amounts ${AMOUNTS} > ${RES_PATH}/${C}_E${E}_QC${QC}_D${D}_QTO${QTO}.txt
            done
        done
    done
done


# Visualization
python visualization/heatmap.py
