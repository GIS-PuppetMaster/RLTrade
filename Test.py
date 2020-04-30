import os
import shutil
from Util.Util import *
from TradeEnv import TradeEnv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def test(save_fig, id, useVersion="final"):
    folder_name, model_path, max_file_name = find_model(id, useVersion)
    # 恢复配置文件
    import yaml
    with open(os.path.join('./wandb', folder_name, 'config.yaml'), 'r') as f:
        conf = f.read()
    conf = yaml.load(conf)
    conf['agent_config'] = conf['agent_config']['value']
    conf['train_env_config'] = conf['train_env_config']['value']
    conf['eval_env_config'] = conf['eval_env_config']['value']
    if conf['train_env_config']['post_processor'] == 'post_processor':
        conf['train_env_config']['post_processor'] = post_processor
    else:
        raise Exception("train_env_config:post_processor为不支持的类型{}".format(conf['train_env_config']['post_processor']))
    if conf['eval_env_config']['post_processor'] == 'post_processor':
        conf['eval_env_config']['post_processor'] = post_processor
    else:
        raise Exception("eval_env_config:post_processor为不支持的类型{}".format(conf['eval_env_config']['post_processor']))
    globals().update(conf)
    globals().update(conf['agent_config'])
    print(model_path)
    model = LoadCustomPolicyForTest(model_path)
    mode = 'test'
    env = TradeEnv(**eval_env_config)
    env.seed(seed)
    env = env.unwrapped
    env.result_path = "E:/运行结果/TRPO/" + folder_name + "/" + mode + "/"
    profit = []
    base = []
    ep = 0
    while ep < n_eval_episodes:
        print(ep)
        s = env.reset()
        flag = False
        for step in range(250):
            a, _ = model.predict(s)
            s, r, done, _ = env.step(a)
            if s is None or r is None:
                flag = True
                break
        env.render("manual")
        if not flag:
            his = np.array(env.trade_history)
            profit_list = np.squeeze(
                (his[:, 4].astype(np.float32) + his[:, 1].astype(np.float32) * his[:, 3].astype(
                    np.float32) - env.principal) / env.principal).tolist()
            price_list = np.array(np.squeeze(his[:, 1]))
            price_list = ((price_list - price_list[0]) / price_list[0]).astype(np.float32).tolist()
            profit.append(profit_list)
            base.append(price_list)
            ep += 1
    # seborn绘图
    plt.close('all')
    ax = plt.subplot(1, 1, 1)
    ax.set_title('TRPO')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving averaged episode averaged profit')
    profit = np.array(profit)
    base = np.array(base)
    sns.tsplot(data=profit, time=np.arange(0, profit.shape[1]), ax=ax, color='r')
    sns.tsplot(data=base, time=np.arange(0, base.shape[1]), ax=ax, color='b')
    if save_fig:
        if not os.path.exists('./TestResult/' + folder_name + '/'):
            os.makedirs('./TestResult/' + folder_name + '/')
        try:
            plt.savefig('./TestResult/' + folder_name + '/' + max_file_name + '.png')
        except:
            print('reward图片被占用，无法写入')
    return plt


if __name__ == "__main__":
    id = "oalhl2pz"
    test(True, id, "final")
    test(True, id, "best")
