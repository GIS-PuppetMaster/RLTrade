import os
import paramiko
from paramiko import SSHClient
from scp import SCPClient

# server_list = [('124.70.7.137', 22), ('124.70.32.29', 22), ('119.3.165.88', 22)]
server_list = [('124.70.32.29', 22)]
abs_path = os.path.split(os.path.abspath(__file__))[:-1][0] + '\\'
files = [('Config\\', ''), ('Env\\', ''), ('Tianshou\\Net\\', 'Tianshou/'), ('Tianshou\\Trainer\\', 'Tianshou/'),
         ('Tianshou\\__init__.py', 'Tianshou/'), ('Tianshou\\SAC.py', 'Tianshou/'),
         ('Tianshou\\StockReplayBuffer.py', 'Tianshou/'), ('Util\\', '')]
files.append(('Tianshou\\TD3.py', 'Tianshou/'))
server_path = '/usr/zkx/Stable-BaselineTrading/'
for ip, port in server_list:
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip,
                port=port,
                username='root',
                password='Hadoop0201')
    scp = SCPClient(ssh.get_transport())
    for file, path in files:
        message = scp.put(abs_path + file, server_path + path, recursive=True)
        print(f'已经更新{file}到{path}')
    ssh.close()
