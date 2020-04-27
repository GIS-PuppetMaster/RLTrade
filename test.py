from stable_baselines import *
from TradeEnv import TradeEnv
import seaborn as sns
import matplotlib.pyplot as plt
from Util.CustomPolicy import *
from Config import *


def test(save_fig):
    stock_code = ['000938_XSHE', '601318_XSHG', '601628_XSHG', '002049_XSHE', '000001_XSHE']
    # exp_name = 'act_fun-gelu_net_arch-vf-256_128_64_32_pi-256_128_64_32_l2_scale-0.01_agent_state-True_episode-50000_EP_LEN-750_n_training_envs-1_GPU-0_save_freq-15000_eval_freq-7500'
    file_list = os.listdir('./checkpoints/' + exp_name)
    max_index = -1
    max_file_name = ''
    for filename in file_list:
        index = int(filename.split("_")[2])
        if index > max_index:
            max_index = index
            max_file_name = filename
    # max_file_name = 'rl_model_24006656_steps.zip'
    model_path = './checkpoints/' + exp_name + '/' + max_file_name
    # model_path = './checkpoints/small_net_5stocks_regularize_StandardScaler/rl_model_97280_steps.zip'
    # model_path = "./BestModels/" + exp_name + "/" + "best_model.zip"
    print(model_path)
    # policy_args = dict(act_fun=gelu)
    # policy_args = dict(act_fun=tf.nn.relu, net_arch=[dict(vf=[256, 128, 64], pi=[64, 64])], l2_scale=0.01)
    model = TRPO.load(model_path, policy_kwargs=policy_args, policy=CustomPolicy)
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
    test(True)
