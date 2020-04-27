from stable_baselines import *
from TradeEnv import TradeEnv
import seaborn as sns
import matplotlib.pyplot as plt
from Util.CustomPolicy import *
from Config import *
import wandb
import os


def test(save_fig, final=True):
    final_path = os.path.join(wandb.run.dir, 'final_model')
    if not final or not os.path.exists(final_path):
        model_path = os.path.join(wandb.run.dir, 'checkpoints')
        file_list = os.listdir(model_path)
        max_index = -1
        max_file_name = ''
        for filename in file_list:
            index = int(filename.split("_")[2])
            if index > max_index:
                max_index = index
                max_file_name = filename
    else:
        model_path = final_path
    print(model_path)
    model = TRPO.load(model_path)
    mode = 'test'
    env = TradeEnv(**eval_env_config)
    env.seed(seed)
    env = env.unwrapped
    env.result_path = "E:/运行结果/TRPO/" + exp_name + "/" + mode + "/"
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
        if not os.path.exists('./TestResult/' + exp_name + '/'):
            os.makedirs('./TestResult/' + exp_name + '/')
        try:
            plt.savefig('./TestResult/' + exp_name + '/' + max_file_name + '.png')
        except:
            print('reward图片被占用，无法写入')
    return plt


if __name__ == "__main__":
    id = ""
    wandb.init(id=id, resume="must")
    test(True)
