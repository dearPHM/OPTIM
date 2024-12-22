import argparse
import os
import pandas as pd
import numpy as np
import random
import math
import pickle
import time


PATH = "./results"


def argparser():
    """ Parse command line arguments for the simulation. """
    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument('--seed', metavar='S', type=int, default=42,
                        help='Seed for simulation')
    parser.add_argument('--repeat', metavar='R', type=int, default=100,
                        help='# simulation')
    parser.add_argument('--stop', metavar='P', type=int, default=2000000,
                        help='Stop this round when for loop reaching P')
    parser.add_argument('--verbose', metavar='V', type=int, default=1,
                        help='0: silent, 1: speak, 2: speak all')
    parser.add_argument('--path', default='./save_simul',
                        help='path for saving pkl files')

    """Blockchain Hyperparams"""

    # January 13, 2023 (https://etherscan.io/chart/blocktime) (https://polygonscan.com/chart/blocktime)
    parser.add_argument('--interval', metavar='I', type=float, default=12.06,
                        help='Average Block Time')
    # January 13, 2023 (https://etherscan.io/chart/tx) (https://polygonscan.com/chart/tx)
    # 154.630975 -> 155, 71.65305 -> 72
    parser.add_argument('--size', metavar='S', type=int, default=155,
                        help='Average # of Transaction per block')
    # Opensea, 2022 Q2 (https://dune.com/queries/690140/1280371) (https://etherscan.io/chart/tx)
    # 5752729 / 99638953 = 0.05773574317 -> 0.0577
    parser.add_argument('--freq', metavar='F', type=float, default=0.0577,
                        help='Training Request / Normal Tx')
    # (0.05 / 155) = 0.0003225806452 : Sereum - Protecting Existing Smart Contracts Against Re-Entrancy Attacks
    # (0.001) : Ethanos - efficient bootstrapping for full nodes on account-based blockchain
    # (0.001) : Speculative Denial-of-Service Attacks in Ethereum
    parser.add_argument('--latency', metavar='L', type=float, default=0.001,
                        help='Latency of normal EVM execution (s)')

    """BRAIN Hyperparams"""

    parser.add_argument('--nodes', metavar='N', type=int, default=21,
                        help='Number of nodes')  # Also, Block Producer (BP)
    parser.add_argument('--byz', metavar='B', type=int, default=0,
                        help='The number of Byzantine nodes')
    parser.add_argument('--epoch', metavar='E', type=int, default=1,
                        help='Epoch [blocks]')
    parser.add_argument('--d', metavar='D', type=int, default=128,
                        help='difficulty (0, 2^256-1], but scaled in (0, 256]')
    parser.add_argument('--qc', metavar='QC', type=int, default=11,
                        help='Quorum of Commitments')
    # parser.add_argument('--qr', metavar='QR', type=int, default=11,
    # help='Quorum of Revelations')
    parser.add_argument('--qto', metavar='O', type=int, default=7881,
                        help='Training Request\'s Timeout [blocks] \ 0 for infinity')
    # parser.add_argument('--tr', metavar='TR', type=int, default=30,
    # help='Period of the Reveal Phase [blocks]')
    # parser.add_argument('--te', metavar='TE', type=int, default=1000,
    # help='Period of the Execute Phase [blocks]')

    parser.add_argument('--times', metavar='T', type=float, default=86400.0,
                        help='The consumed time of training')
    parser.add_argument('--amounts', metavar='A', type=int, default=1000,
                        help='The amount of training requests')

    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    if args.verbose > 1:
        print()
        print(args)
        print("=" * 105)
        print()

    # Hyp
    n_qtx = args.amounts
    times = [args.times for _ in range(n_qtx)]  # sec

    if args.verbose > 0:
        print(f"# Training Requests: {n_qtx}")
        # print(f"Normal TX latency: {args.latency:8.4f}, Avg Training latency: {np.average(times[np.nonzero(times)]):8.4f}, (SD: {times.std():8.4f})")
        print(f"All Training time: {sum(times):10.4f}")
        print(
            f"Avg Training time: {np.average(times[np.nonzero(times)]):10.4f}, (SD: {times.std():10.4f})")
        print("=" * 105)
        print()

    """Simulator"""

    # (priority, (block_height_when_req, time, qtxid))
    from queue import PriorityQueue
    # top() -> qs.queue[0]
    # pop() -> get()
    # push() -> put()

    def pareto(size, alpha=1.16, lower=0., upper=1.):
        s = np.random.pareto(alpha, size)
        s /= sum(s)
        s *= (upper - lower)
        s += lower
        return s.clip(lower, upper)

    repeat = args.repeat
    nodes = args.nodes

    # Evaluation data above all rounds
    latencies_block = np.array([], dtype=int)
    stales_block = np.array([], dtype=int)
    blocks = list()
    n_txs = list()
    timeouts = list()
    additional_txs = list()
    max_queue_lens = list()

    for r in range(repeat):  # in this repeat,
        random.seed(args.seed + r)  # fix seed for each round

        # BRAIN
        qs = PriorityQueue()  # lower is first. (because of pareto dist.)
        # 0 for highest priority, 1 for secondary
        ps = pareto(n_qtx, lower=2, upper=1000)
        commitments = np.zeros(n_qtx)
        # revelations = np.array([0 for _ in range(n_qtx)])  # assumption: all nodes reveal commitment in an one block.
        committee = np.zeros((n_qtx, args.nodes))

        # Metrics
        # latency = trainings[id][end] - trainings[id][start] in [Block]  # for evaluation
        trainings = [dict({'start': -1, 'end': -1, 'stale': None})
                     for _ in range(n_qtx)]

        # create tx set
        txn = int(n_qtx / args.freq)
        # -1 for normal txs, 0~N for training queries
        txs = [-1 for _ in range(txn)] + [t for t in range(n_qtx)]
        random.shuffle(txs)
        txs += [-1 for _ in range(args.stop - len(txs))]

        # for VRF
        # seed = args.seed + r

        # data
        current_block = 0
        timeout_count = 0
        training_count = 0
        additional_tx = 0
        max_queue_len = 0

        cached_y = [False for _ in range(nodes)]

        for txid, tx in enumerate(txs):  # each tx w/ txid
            if txid % args.size == 0:
                current_block += 1

            if args.verbose > 1:
                print(f"Round {r:4d}, Block {current_block:4d}, Training: # {training_count:4d}, Timeout: # {timeout_count:4d}, txid {txid:6d} / {len(txs):6d}, tx {'normal' if tx == -1 else 'trainq'} {tx:4d}", end='\r')

            # Tx
            if tx == -1:  # normal tx
                # change Epoch: clear cache
                if current_block % args.epoch == 0:
                    cached_y = [False for _ in range(nodes)]
            else:  # training request tx
                # change Epoch: clear cache
                cached_y = [False for _ in range(nodes)]

                # Fail case
                if (args.nodes - args.byz) < args.qc:
                    timeout_count += 1
                    continue

                # training request, push()
                qs.put((ps[tx], (current_block, times[tx], tx)))
                trainings[tx]['start'] = current_block
                trainings[tx]['stale'] = 0
                # counting on tx - no additional tx

                if qs.qsize() > max_queue_len:
                    max_queue_len = qs.qsize()

            # Each node
            for n in range(nodes):
                if qs.empty():
                    pass  # nothing in the queue
                elif qs.queue[0][0] != 0:  # nothing to refer now
                    top = qs.queue[0]  # top
                    requested_block = top[1][0]
                    if requested_block < current_block:
                        q = qs.get()[1]  # update
                        # update w/ highest priority. Now can refer it.
                        qs.put((0, (requested_block, q[1], q[2])))

                # Actions: not empty, highest priority.
                if (not qs.empty()) and (qs.queue[0][0] == 0):
                    # Timeout
                    if (current_block - qs.queue[0][1][0] >= args.qto) if args.qto > 0 else False:
                        qs.get()
                        timeout_count += 1
                        # break

                    # Enough block height: Do action.
                    elif (qs.queue[0][1][0] < current_block):
                        q = qs.queue[0][1]

                        # vrf
                        # vrf_seed = int(f"{q[2]}{seed}{int(current_block / args.epoch)}{n}")
                        # random.seed(vrf_seed)
                        if not cached_y[n]:
                            cached_y[n] = random.randint(0, 256)

                        # byzantine
                        # probabilistic approach
                        bp = random.randint(0, args.nodes)

                        # join committee
                        if (bp >= args.byz) and (cached_y[n] <= args.d) and (committee[q[2]][n] == 0):
                            # training
                            # print(f"- node {n} do training")

                            # commit
                            commitments[q[2]] += 1
                            additional_tx += 2  # commit and reveal
                            committee[q[2]][n] = 1

                            if commitments[q[2]] >= args.qc:
                                # seed update
                                # random.seed(seed)
                                # seed = random.randint(0, 256)  # 0~256

                                # pop
                                qs.get()
                                # for training time
                                trainings[q[2]]['end'] = current_block + \
                                    int(times[q[2]] / args.interval) + 1
                                training_count += 1
                                current_block += 1

                                # update stale
                                for i, training in enumerate(trainings):
                                    if (training['stale'] != None) and (training['end'] == -1):
                                        training['stale'] += 1

                        elif (bp < args.byz) and (n == args.nodes - 1):  # byzantine case
                            current_block += 1

            if (training_count + timeout_count) == n_qtx:
                break

        """End of Round"""
        if args.verbose == 1:
            print(f"Round {r:4d} End", end='\r')

        # print("=" * 105)
        for i, training in enumerate(trainings):
            if training['end'] == -1:
                # TODO: fallbacks (for not ended qtxs)
                if args.verbose > 1:
                    print(f"Fallbacks @ {r:4d}, {i:4d}, {training}")
            else:
                latencies_block = np.append(
                    latencies_block, [training['end'] - training['start']])  # latencies_block
                stales_block = np.append(
                    stales_block, [training['stale']])

        if args.verbose > 1:
            print(
                f"Round {r:4d}, Block {current_block:4d}, Training: # {training_count:4d}, Timeout: # {timeout_count:4d}" + " " * 19)
            print("=" * 105)
            print()
        # print("Queue", qs.queue)
        # Per round operations
        blocks.append(current_block + 1)  # +1
        n_txs.append(math.ceil(txid / args.size) * args.size)
        timeouts.append(timeout_count)
        max_queue_lens.append(max_queue_len)
        additional_txs.append(additional_tx)

    # All round operations
    # print("=" * 105)
    # print("Blocks per round:", blocks)
    # print("TXs per round     :", n_txs)

    def evaluate(A):
        if len(A) == 0:
            return "Empty"
        return f"Min {min(A):10.4f}, Max {max(A):10.4f}, Avg {(np.average(A)):10.4f} (SD: {A.std():10.4f}), MED {(np.median(A)):10.4f}"

    blocks = np.array(blocks)
    n_txs = np.array(n_txs)  # number of tasks
    timeouts = np.array(timeouts)  # failed training tasks
    max_queue_lens = np.array(max_queue_lens)
    additional_txs = np.array(additional_txs)

    BRAIN_elapsed_times = (n_txs + additional_txs) * args.latency
    others_elapsed_times = (n_txs - n_qtx) * args.latency + sum(times)

    print(f"Executed Time (s)       :", evaluate(BRAIN_elapsed_times))
    print(f"Executed Training Tasks :", evaluate(n_qtx - timeouts))
    print(f"Timeout  Training Tasks :", evaluate(timeouts))
    print()
    print(f"TPS (tx/s) No Training  :", evaluate(np.ones(args.repeat)
          * (1 / args.latency)))  # tasks / (tasks * per_latency)
    print(f"TPS (tx/s) Other        :", evaluate(n_txs / others_elapsed_times))
    print(f"TPS (tx/s) BRAIN        :", evaluate(n_txs /
          BRAIN_elapsed_times))  # [Task Per Second]
    print()
    print(f"Max Queue Length        :", evaluate(max_queue_lens))
    print(f"Latency [blocks]        :", evaluate(latencies_block))
    print(f"Stale [blocks]          :", evaluate(stales_block))
    if args.verbose > 0:
        print("=" * 105)
        print()

    file_name = f'{args.path}/objects/R{args.repeat}_F{args.freq}_N{args.nodes}_B{args.byz}_E{args.epoch}_D{args.d}_QC{args.qc}_T{args.times}_A{args.amounts}_{time.time()}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([
            # Data
            n_txs, n_qtx, additional_txs, timeouts,
            BRAIN_elapsed_times, others_elapsed_times,
            # TPS
            np.ones(args.repeat) * (1 / args.latency),
            n_txs / others_elapsed_times,
            n_txs / BRAIN_elapsed_times,
            # Latency
            max_queue_lens, latencies_block, stales_block], f)
