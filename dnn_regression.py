import random
import scipy.io
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def load_data(collect_time=1000, shuffle_flag=True):
    # m = scipy.io.loadmat('./flight_data.mat')
    # data_raw = m['flight_data'][0, 0]['dnn0']
    # for collect_i in range(1, collect_time):
    #     data_raw = np.concatenate((data_raw,  m['flight_data'][0, 0]['dnn{}'.format(collect_i)]), axis=0)
    m = scipy.io.loadmat('./flight_data_concatenate.mat')
    data_raw = np.array(m['data_raw'])
    data_raw = np.true_divide(data_raw, [1e2, -1, 1e4, -1, 1e4, 1e4, 1])
    print(np.mean(data_raw, axis=0))
    if shuffle_flag:
        np.random.shuffle(data_raw)
    size_raw = len(data_raw)
    size_train = round(size_raw * 0.96)
    x_train_set = data_raw[:size_train, :-1]
    y_train_set = data_raw[:size_train, -1:]

    size_test = size_train + round(size_raw * 0.02)
    x_test_set = data_raw[size_train:size_test, :-1]
    y_test_set = data_raw[size_train:size_test, -1:]

    x_validation_set = data_raw[size_test:, :-1]
    y_validation_set = data_raw[size_test:, -1:]
    print('data load DONE! Train data size=', size_train)
    return x_train_set, y_train_set, x_test_set, y_test_set, x_validation_set, y_validation_set, size_train


def init_network(x_num=6, learn_rate=0.0001):
    with tf.name_scope("placeholder"):
        x = tf.placeholder(tf.float32, shape=[None, x_num], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    with tf.name_scope("layers"):
        l1 = x
        for i in range(3):
            l1 = tf.layers.dense(l1, units=100, activation=tf.nn.relu, name="n_hidden_" + str(i))
        y_hat = tf.layers.dense(l1, units=1, name="n_outputs")
    with tf.name_scope("loss"):
        loss_func = tf.reduce_mean(tf.square(y_hat - y), name="loss")
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        trainer = optimizer.minimize(loss_func)
    return x, y, trainer, loss_func


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    x, y, train_step, loss_mse = init_network()

    x_train, y_train, x_test, y_test, x_vali, y_vali, train_set_size = load_data()

    load_flag = True

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if load_flag is True:
        checkpoint_path = tf.train.latest_checkpoint('./dnn_model')
        saver.restore(sess, checkpoint_path)
        print("test loss mse = %g" % loss_mse.eval(feed_dict={x: x_test, y: y_test}))

    steps = int(1e5)
    batch_size = int(1e4)
    for i in range(steps):
        # batch = ([], [])
        # p = random.sample(range(train_set_size), batch_size)
        # for k in p:
        #     batch[0].append(x_train[k])
        #     batch[1].append(y_train[k])
        # train_step.run(feed_dict={x: batch[0], y: batch[1]})
        p = int(random.uniform(0, train_set_size - batch_size))
        train_step.run(feed_dict={x: x_train[p:p + batch_size, :], y: y_train[p:p + batch_size, :]})
        if i % (steps / 10) == 0:
            train_loss_mse = loss_mse.eval(feed_dict={x: x_test, y: y_test})
            print('Training...{:.1f}%'.format(i / steps * 100), 'train loss mse =', train_loss_mse)
    print("test loss mse = %g" % loss_mse.eval(feed_dict={x: x_vali, y: y_vali}))

    saver.save(sess, r'.\dnn_model\flight')
