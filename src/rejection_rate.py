from xml.etree.ElementInclude import include
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

matplotlib.use('Agg')

# accuracy
for resource in os.scandir("save/rejection"):
    if resource.is_dir and resource.name.find("accuracy") != -1:
        args = resource.name

        acc_grad_arr = [[], []]
        acc_norej_arr = [[], []]
        acc_optim_arr = [[], []]

        for file_name in os.listdir(resource.path + "/objects"):
            if file_name.find("brain") != -1:
                with open(resource.path + "/objects/" + file_name, 'rb') as file:
                    loaded_arr = pickle.load(file)

                    for arr in loaded_arr:
                        while (len(arr) > 200):
                            arr.pop(-1)

                    iid_idx = 0
                    if file_name.find("Non") != -1:
                        iid_idx = 1

                    if file_name.find("H0_") != -1:
                        acc_optim_arr[iid_idx].append(loaded_arr[1])
                    elif file_name.find("H2_") != -1:
                        acc_norej_arr[iid_idx].append(loaded_arr[1])
                    else:
                        acc_grad_arr[iid_idx].append(loaded_arr[1])

        for i in range(2):
                acc_grad_arr[i] = np.mean(acc_grad_arr[i], axis=0)
                acc_optim_arr[i] = np.mean(acc_optim_arr[i], axis=0)
                acc_norej_arr[i] = np.mean(acc_norej_arr[i], axis=0)

        print(args)
        # print("acc diff: ", np.mean(acc_optim - acc_grad))

        # print("            brain\toptim")
        print("avg acc - iid: {:.3f}  {:.3f} ".format(np.mean(np.hsplit(np.array(acc_optim_arr[0]),2)[1]), np.mean(np.hsplit(np.array(acc_grad_arr[0]),2)[1])))
        print("avg acc - non iid: {:.3f}  {:.3f} ".format(np.mean(np.hsplit(np.array(acc_optim_arr[1]),2)[1]), np.mean(np.hsplit(np.array(acc_grad_arr[1]),2)[1])))
        print(f"std - iid:  {np.std(np.hsplit(np.array(acc_optim_arr[0]),2)[1])}        {np.std(np.hsplit(np.array(acc_grad_arr[0]),2)[1])}")
        print(f"std - non iid:  {np.std(np.hsplit(np.array(acc_optim_arr[1]),2)[1])}        {np.std(np.hsplit(np.array(acc_grad_arr[1]),2)[1])}")
        # Plot Average Accuracy vs Communication rounds
        plt.figure(figsize=(9,5))
        # plt.title('Average Accuracy vs Communication rounds')
        plot_iid = plt.subplot(1, 2, 1)
        plot2 = plt.plot(range(200), acc_norej_arr[0], color='black', linestyle = '--', label='OPTIM without consecutive rejection')
        plt.setp(plot2, color='black', linewidth=1.0)
        plot1 = plt.plot(range(200), acc_optim_arr[0], color='b', label='OPTIM')
        plt.setp(plot1, color='b', linewidth=1.0)
        plot0 = plt.plot(range(200), acc_grad_arr[0], color='orange', label='OPTIM with Gradual')
        plt.setp(plot0, color='orange', linewidth=1.0)
        
        plt.legend()
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds\n\nIID')

        plot_non_iid = plt.subplot(1, 2, 2, sharey=plot_iid)
        plot2 = plt.plot(range(200), acc_norej_arr[1], color='black', linestyle = '--', label='OPTIM without consecutive rejection')
        plt.setp(plot2, color='black', linewidth=1.0)
        plot1 = plt.plot(range(200), acc_optim_arr[1], color='b', label='OPTIM')
        plt.setp(plot1, color='b', linewidth=1.0)
        plot0 = plt.plot(range(200), acc_grad_arr[1], color='orange', label='OPTIM with Gradual')
        plt.setp(plot0, color='orange', linewidth=1.0)
        plt.yticks(visible=False)
        plt.xlabel('Communication Rounds\n\nnon-IID')

        plt.subplots_adjust(left=0.15)        

        plt.legend()
        plt.tight_layout()
        plt.savefig(resource.path + '/{}_{}.png'.format(args, 'acc'))
        os.chmod(resource.path + '/{}_{}.png'.format(args, 'acc'), 0o777)

        plt.close('all')
    
    elif resource.is_dir and resource.name.find("commit") != -1:

        rejection = [[[], []], [[], []]]
        total = [[[], []], [[], []]]

        for file_name in os.listdir(resource.path + "/objects"):
            with open(resource.path + "/objects/" + file_name, 'rb') as file:
                if file_name.endswith(".png") or file_name.find("H2_") != -1:
                    continue

                arrs = pickle.load(file)

                for arr in arrs:
                    for i in range(1, 200):
                        arr[i] += arr[i-1]

                    while len(arr) > 200:
                        arr.pop(-1)

                iid_idx = 0
                method_idx = 0

                if file_name.find("H1_") != -1:
                    method_idx = 1
                
                if file_name.find("Non") != -1:
                    iid_idx = 1

                v, r, rr = arrs

                t = np.add(v, np.add(r, rr))

                rejection[iid_idx][method_idx].append(rr)
                total[iid_idx][method_idx].append(t)
        
        for i in range(2):
            for j in range(2):
                rejection[i][j] = np.mean(rejection[i][j], axis=0)
                total[i][j] = np.mean(total[i][j], axis=0)

        [rate_iid_optim, rate_iid_grad, rate_noniid_optim, rate_noniid_grad] = [np.divide(rejection[i][j], total[i][j]) for j in range(2) for i in range(2)]

        # rate graph
        plt.figure()
        plot0 = plt.plot(range(200), rate_noniid_optim, color='b', label='OPTIM')
        plt.setp(plot0, color='b', linewidth=1.0)

        plot1 = plt.plot(range(200), rate_noniid_grad, color='c', label='OPTIM with Gradual')
        plt.setp(plot1, color='c', linewidth=1.0)
        plt.subplots_adjust(left=0.15)
        plt.ylabel('Average Rejection Rate')
        plt.xlabel('Communication Rounds')
        plt.legend()
        plt.savefig('save/rejection/commit/rate.png')
        os.chmod('save/rejection/commit/rate.png', 0o777)
        print(rejection[0][0][-1])
        print(rejection[1][0][-1])
        #rejection count
        plt.figure()
        plot0 = plt.plot(range(200), rejection[1][0], color='b', label='OPTIM')
        plt.setp(plot0, color='b', linewidth=1.0)

        plot1 = plt.plot(range(200), rejection[1][1], color='c', label='OPTIM with Gradual')
        plt.setp(plot1, color='c', linewidth=1.0)
        plt.subplots_adjust(left=0.15)
        plt.ylabel('Average Rejection Count')
        plt.xlabel('Communication Rounds')
        plt.legend()
        plt.savefig('save/rejection/commit/count.png')
        os.chmod('save/rejection/commit/count.png', 0o777)

        # rate bar
        plt.figure()
        plt.bar(np.arange(1), [rate_noniid_optim[-1]], color='b', label='OPTIM')
        plt.bar(np.arange(1) + 1, [rate_noniid_grad[-1]], color='c', label='OPTIM with Gradual')

        plt.subplots_adjust(left=0.15)
        plt.ylabel('Average Rejection Rate')
        plt.xlabel('dataset')
        plt.ylim([0, 1])

        plt.xticks(np.arange(1) + 1/2,['iid']) 
        plt.legend()
        plt.savefig('save/rejection/commit/rate_bar.png')
        os.chmod('save/rejection/commit/rate_bar.png', 0o777)

        # rate bar2
        plt.figure(figsize=(6,2.5))

        y = np.arange(3)
        labels = ['OPTIM with Gradual, non-IID','OPTIM with Gradual, IID', 'OPTIM']
        values = [(rejection[1][1][-1] / rejection[1][0][-1]) * 100, (rejection[0][1][-1] / rejection[0][0][-1]) * 100, 100]
        barh = plt.barh(y, values, color=['orange', 'orange', 'b'], height=0.7)
        # plt.barh(np.arange(1) + 1, [rejection[0][1] / rejection[0][0]], color='c', label='OPTIM with Gradual - iid')
        # plt.barh(np.arange(1) + 1, [rejection[1][1] / rejection[1][0]], color='c', label='OPTIM with Gradual - iid')
        plt.yticks(y, labels)
        plt.xlim((0, 135))
        plt.xticks([], [])

        xpad = 0.35
        plt.subplots_adjust(left=xpad)
        for idx, bar in enumerate(barh):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() + 4, bar.get_y() + 0.28, f'{values[idx]:.1f} %', ha='left', size = 12)
        
        plt.axis("off")

        plt.savefig('save/rejection/commit/rate_bar2.png')
        os.chmod('save/rejection/commit/rate_bar2.png', 0o777)

        plt.close('all')
