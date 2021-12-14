import os
import threading
import numpy as np

class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self,type,value,traceback): pass

def set_session(sess): pass

graph = DummyGraph()
sess = None

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    print('Eager Mode: {}'.format(tf.executing_eagerly()))
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim = 0, output_dim = 0, lr = 0.001, shared_network = None, activation = 'sigmoid', loss = 'mse'):
        self.input_dim = input_dim  #입력 데이터
        self.output_dim = output_dim #출력 데이터
        self.lr = lr  #학습 속도
        self.shared_network = shared_network  #공유 신경망
        self.activation = activation  #활성화 함수
        self.loss = loss #학습 손실
        self.model = None #최종 신경망 모델

# 샘플에대한 행동의 가치 또는 확률 예측
    def predict(self,sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

#학습 데이터와 레이블 x,y를 입력으로 받아서 모델을 학습
#A3C에서는 여러 스레드가 병렬로 신경망을 사용할 수 있기때문에 충돌이 일어나지 않게 스레드들의 동시 사용을 막습니다.
    def train_on_batch(self,x,y):
        loss = 0.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x,y)
        return loss
#모델을 파일로 저장 함수
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weight(model_path, overwrite =True)  #HDF5 파일 저장

#파일로부터 모델을 읽어오는 함수
    def load_model(self,model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

#클래스함수는 인스턴스를 만들지 않고 사용할 수 있는 함수/ className.function_name()과 같이 호출, 유의할 점은 클래스 함수에서는 인스턴스 변수인 self.variable_name을 사용할 수 없다.
#DNN, LSTM, CNN 신경망의 공유 신경망을 생성하는 클래스 함수
    @classmethod
    def get_shared_network(cls,net='dnn', num_steps = 1, input_dim = 0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lsm':
                return LSTMNetwork.get_network_head(Input((num_steps, input_dim)))
            elif net == 'cnn':
                return CNN.get_network_head(Input((1,num_steps, input_dim)))

class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)

            inp = None
            output = None
            if self.shared_network is None:   #없으면 공유 신경망 생성성
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.get_network.output

            output = Dense(self.output_dim, activation = self.activation, kernel_initializer='random_normal')(output)

            self.model - Model(inp, output)
            self.model_compile(optimizer = SGD(lr=self.lr),loss = self.loss)


    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)  #배치 정규화로 학습을 안정화, 배치 정규화는 은닉 레이어의 입력을 정규화해 학습을 가속화하는 방법
        output = Dropout(0.1)(output)  #과적합을 일정 부분 피함
        output = Dense(128, activation='sigmoid', kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid',  kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid',  kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)


    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)


    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)




class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        # cuDNN 사용을 위한 조건
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        output = LSTM(256, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1, stateful=False, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim, 1))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),   #padding 옵션same은 입력과 출력의 크기를 같게 설정
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)

