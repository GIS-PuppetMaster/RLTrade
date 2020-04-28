from stable_baselines import *
from TradeEnv import TradeEnv
import seaborn as sns
import matplotlib.pyplot as plt
from Config import *
import os
from Util.CustomPolicy import LoadCustomPolicyForTest

def find_model(id, useVersion="final", main_path="./"):
    if id is None or id == "":
        raise ("id不能为空")
    fl = os.listdir('./wandb/')
    folder_name = None
    for file in fl:
        ID = file.split("-")[-1]
        if id == ID:
            folder_name = file
    if folder_name is None:
        raise ("未找到包含id:{}的文件夹".format(id))
    if useVersion == "last" or (useVersion == 'final' and not os.path.exists(
            os.path.join(main_path, 'wandb', folder_name, 'final_model.zip'))):
        model_path = os.path.join(main_path, 'wandb', folder_name, 'checkpoints/')
        file_list = os.listdir(model_path)
        max_index = -1
        max_file_name = ''
        for filename in file_list:
            index = int(filename.split("_")[2])
            if index > max_index:
                max_index = index
                max_file_name = filename
        model_path = os.path.join(model_path, max_file_name)
    elif useVersion == "final":
        model_path = os.path.join(main_path, 'wandb', folder_name, 'final_model.zip')
        max_file_name = "final"
    elif useVersion == "best":
        model_path = os.path.join(main_path, 'wandb', folder_name, 'best_model.zip')
        max_file_name = "best_model"
    else:
        model_path = os.path.join(main_path, 'wandb', folder_name, 'checkpoints', useVersion)
        max_file_name = useVersion
    return folder_name, model_path, max_file_name


def test(save_fig, id, useVersion="final"):
    folder_name, model_path, max_file_name = find_model(id, useVersion)
    # folder_name = 'oldEnvOldCheckpoint'
    # max_file_name = 'rl_model_25589760_steps.zip'
    # model_path = './checkpoints/small_net_5stocks_regularize_StandardScaler/'+max_file_name
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
    id = "qdpulf2y"
    test(True, id, "final")
    test(True, id, "best")
