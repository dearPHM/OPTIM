import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

matplotlib.use('Agg')

res_list = []

for resource in os.scandir("save/"):
    res_list.append(resource)

res_list.sort(key= lambda x: x.name)

for resource in res_list:
    if resource.is_dir and resource.name.find("brain") != -1 and resource.name.find("H2") == -1:
        args = resource.name

        acc_brain_arr = []
        acc_optim_arr = []

        loss_brain_arr = []
        loss_optim_arr = []

        min_len = 1024

        for file_name in os.listdir(resource.path + "/objects"):
            if file_name.find("brain") != -1:
                with open(resource.path + "/objects/" + file_name, 'rb') as file:
                    loaded_arr = pickle.load(file)

                    min_len = min(len(loaded_arr[0]), min_len)

                    if file_name.find("brain_optim") == -1:
                        acc_brain_arr.append(loaded_arr[1])
                        loss_brain_arr.append(loaded_arr[0])
                    else:
                        acc_optim_arr.append(loaded_arr[1])
                        loss_optim_arr.append(loaded_arr[0])

        for arr in [acc_brain_arr, loss_brain_arr, acc_optim_arr, loss_optim_arr]:
            for it in arr:
                while (len(it) > 200):
                    it.pop(-1)

        [acc_brain, loss_brain, acc_optim, loss_optim] = [np.mean(arr, axis=0) for arr in [acc_brain_arr, loss_brain_arr, acc_optim_arr, loss_optim_arr]]

        b_75, b_80, o_75, o_80 = [-1,-1,-1,-1]

        print(args)
        print("acc diff: ", np.mean(acc_optim - acc_brain))

        for idx, acc in enumerate(acc_brain):
            if (acc >= 0.75 and b_75 < 0):
                b_75 = idx
            if (acc >= 0.80 and b_80 < 0):
                b_80 = idx
        for idx, acc in enumerate(acc_optim):
            if (acc >= 0.75 and o_75 < 0):
                o_75 = idx
            if (acc >= 0.80 and o_80 < 0):
                o_80 = idx

        print("            brain\toptim")

        print(f"std:  {np.std(np.hsplit(np.array(acc_brain_arr),2)[1])}        {np.std(np.hsplit(np.array(acc_optim_arr),2)[1])}")
        # print(f"std:  {np.std(acc_brain)}        {np.std(acc_optim)}")
        print("avg acc: {:.3f}      {:.3f} ".format(np.mean(acc_brain), np.mean(acc_optim)))
        print("peak acc: {:.3f}     {:.3f} ".format(np.max(acc_brain), np.max(acc_optim)))
        print("over 75: ", b_75, '      \t', o_75)
        print("over 80: ", b_80, '      \t', o_80, '\n')


        # Plot Loss curve
        plt.figure()
        plt.title('Average Validation Loss vs Communication rounds')
        plot0 = plt.plot(range(200), loss_brain, color='r', label='Original')
        plt.setp(plot0, color='r', linewidth=1.0)

        plot1 = plt.plot(range(200), loss_optim, color='b', label='Optim')
        plt.setp(plot1, color='b', linewidth=1.0)
        plt.subplots_adjust(left=0.15)
        plt.ylabel('Average Validation loss')
        plt.xlabel('Communication Rounds')
        plt.legend()
        plt.savefig(resource.path + '/{}_{}.png'.format(args, 'loss'))
        os.chmod(resource.path + '/{}_{}.png'.format(args, 'loss'), 0o777)

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        # plt.title('Average Accuracy vs Communication rounds')
        plot0 = plt.plot(range(200), acc_brain, color='r', label='Original')
        plt.setp(plot0, color='r', linewidth=1.0)

        plot1 = plt.plot(range(200), acc_optim, color='b', label='OPTIM')
        plt.setp(plot1, color='b', linewidth=1.0)
        plt.subplots_adjust(left=0.15)
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.legend()
        plt.savefig(resource.path + '/{}_{}.png'.format(args, 'acc'))
        os.chmod(resource.path + '/{}_{}.png'.format(args, 'acc'), 0o777)

        plt.close('all')
