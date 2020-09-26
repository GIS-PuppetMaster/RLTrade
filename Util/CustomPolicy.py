import gym
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn, MlpPolicy, \
    FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        seed = 0
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            net_arch = [dict(vf=[256, 128, 64, 32], pi=[256, 128, 64, 32])]
            l2_scale = 0.01
            dropout_rate = 0.0
            training = False
            for k, v in kwargs.items():
                if k == 'act_fun':
                    activ = v
                elif k == 'net_arch':
                    net_arch = v
                elif k == 'l2_scale':
                    l2_scale = v
                elif k == 'dropout_rate':
                    dropout_rate = v
                    if dropout_rate == 0.:
                        training = False
                    else:
                        training = True
                else:
                    raise Exception("不支持的Policy_arg:{}".format(k))
            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            extracted_features = self.processed_obs
            if len(ob_space.shape) > 1:
                extracted_features = tf.layers.flatten(extracted_features)
            index_code = 0
            arch_pi_and_vf = None
            for arch in net_arch:
                if isinstance(arch, int):
                    extracted_features = tf.layers.dropout(activ(
                        tf.layers.dense(extracted_features, arch, name='feature_extract' + str(arch),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))),
                        rate=dropout_rate,
                        seed=seed,
                        training=training)
                    index_code += 1
                elif isinstance(arch, dict):
                    arch_pi_and_vf = arch
                else:
                    raise Exception("自定义网络参数不合法: {}".format(arch))
            pi_layer_size = arch_pi_and_vf['pi']
            vf_layer_size = arch_pi_and_vf['vf']
            pi_h = extracted_features
            for i, layer_size in enumerate(pi_layer_size):
                pi_h = tf.layers.dropout(activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i),
                                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                   l2_scale))), rate=dropout_rate, seed=seed,
                                         training=training)
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate(vf_layer_size):
                vf_h = tf.layers.dropout(activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i),
                                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                   l2_scale))), rate=dropout_rate, seed=seed,
                                         training=training)
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CustomMultiStockPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomMultiStockPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                     scale=True)

        with tf.variable_scope("model", reuse=reuse):
            extracted_features = self.processed_obs
            batch, time, stocks, feature = extracted_features.shape
            convert = lambda x: x.value
            time, stocks, feature = convert(time), convert(stocks), convert(feature)
            extracted_features = tf.reshape(extracted_features, shape=(-1, time, stocks * feature))
            extracted_features = tf.nn.leaky_relu(
                tf.layers.conv1d(extracted_features, 32, 9, dilation_rate=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            extracted_features = tf.nn.leaky_relu(tf.layers.conv1d(extracted_features, 32, 7, dilation_rate=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            extracted_features = tf.nn.leaky_relu(tf.layers.conv1d(extracted_features, 64, 5, dilation_rate=4, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            extracted_features = tf.nn.leaky_relu(tf.layers.conv1d(extracted_features, 128, 3, dilation_rate=8, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            extracted_features = tf.nn.leaky_relu(tf.layers.conv1d(extracted_features, 256, 8, dilation_rate=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            extracted_features = tf.layers.flatten(extracted_features)
            pi_h = extracted_features
            for i, layer_size in enumerate([256, 128]):
                pi_h = tf.nn.leaky_relu(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i), kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([256, 128]):
                vf_h = tf.nn.leaky_relu(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i), kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            value_fn = tf.nn.leaky_relu(tf.layers.dense(vf_h, 1, name='vf_fc'+str(i+1), kernel_regularizer=tf.contrib.layers.l2_regularizer(1.)))
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
