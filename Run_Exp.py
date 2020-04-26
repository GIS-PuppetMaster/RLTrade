import os
import shutil
import time
import wandb
from multiprocessing import Process


def run():
    os.system('python TRPO.py')


if __name__ == '__main__':
    config_list = os.listdir('ConfigSet')
    wandb.login()
    for config_file in config_list:
        shutil.copyfile(os.path.join('./ConfigSet', config_file), './Config.py')
        start_time = time.time()
        while not os.path.exists('./Config.py'):
            if time.time() - start_time >= 30:
                raise ("缺少Config.py")
            pass
        Process(target=run).start()
        time.sleep(10)
        os.remove('./Config.py')
    print("启动完毕")
