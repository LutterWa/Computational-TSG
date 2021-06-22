import scipy.io
import random
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from dnn_regression import load_data, init_network


class DeepNeuralNetwork:
    def __init__(self, x_num=6, path='./dnn_model'):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x, y, trainer, loss = init_network(x_num=x_num)
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.y = self.sess.graph.get_tensor_by_name(name='layers/n_outputs/BiasAdd:0')
                saver = tf.train.Saver()
                checkpoint_path = tf.train.latest_checkpoint(path)
                saver.restore(self.sess, checkpoint_path)

                # 转换为tf-lite模型
                convert = tf.lite.TFLiteConverter.from_session(self.sess, [self.x], [self.y])
                tflite_model = convert.convert()
                with open('dnn_model.tflite', 'wb') as f:
                    f.write(tflite_model)

    def predict(self, state):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                return self.y.eval(feed_dict={self.x: state})


if __name__ == '__main__':
    dnn = DeepNeuralNetwork()
    x_train, y_train, x_test, y_test, x_vali, y_vali, train_set_size = load_data(shuffle_flag=False)
    # p = int(random.uniform(0, train_set_size - 1e6))
    dict_DNN = {'y_hat': dnn.predict(x_test), 'y': y_test}
    dataDNN = r'dnn_test.mat'
    scipy.io.savemat(dataDNN, dict_DNN)
