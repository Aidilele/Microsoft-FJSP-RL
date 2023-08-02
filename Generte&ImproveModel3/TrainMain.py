from GIModel import *
import numpy as np


def main():
    update_model_dict = {1: 'Improve', -1: 'Generate'}
    gim = GIM(config_path="./config.json")
    gim.load()
    ave_cmax = 0
    ep_count = 0
    print_count = 0
    updata_model_flag = 1
    for episode in range(1,gim.config["CommonTrainParas"]["max_episodes"] + 1):
        end_cmax = gim.model[gim.config["CommonTrainParas"]["train_model"]]()
        gim.update(repeat=5, model=update_model_dict[1])
        ave_cmax += end_cmax
        ep_count += 1
        if episode % 500 == 0:
            gim.save(episode)
        if episode % 2000 == 0:  # 变换更新参数的模型
            updata_model_flag *= -1
        if ep_count % len(gim.ie.graph_list) == 0:
            ave_cmax = ave_cmax / len(gim.ie.graph_list)
            gim.summer.add_scalar("EP Cmax ", ave_cmax, print_count)
            print(
                "Episode : {} \t\t EP Cmax : {:6.2f}".format(
                    ep_count,
                    ave_cmax,
                ))
            print_count += 1
            ave_cmax = 0


if __name__ == '__main__':
    main()
