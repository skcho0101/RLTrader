import os  #폴더 생성이나 파일 경로 준비
import logging
import abc   #추상 클래스 정의
import collections
import threading
import time
import numpy as np
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    # stock_code 학습을 진행하는 주식 종목
    # chart_data 강화 학습의 환경에 해당하는 주식 일봉 차트 데이터
    # trading_data - 학습을 위한 천처리된 학습 데이터
    # delayed_reward_threshold - 지연 보상 임계값, 수익율이나 손실률이 이 임계값보다 클 경우 지연 보상이 발생해 이전 행동들에 대한 학습이 진행됩니다.
    # lr - 학습 속도로 이 값이 너무 크면 학습이 제대로 진행되지 않으며 너무 작으면 학습이 너모 오래 걸림

    def __init__(self, rl_method = 'r1', stock_code = None,
                 chart_data = None, training_data = None,
                 min_trading_unit = 1, max_trading_unit=2,
                 delayed_reward_threshold = .05,
                 net = 'dnn', num_steps = 1, lr = 0.001,
                 value_network = None, policy_network = None,
                 output_path = '',reuse_models = True):
        #인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        #강화 학습 기법 설정
        self.rl_method = rl_method
        #환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.enviroment = Environment(chart_data)
        #에이전트 설정
        self.agent = Agent(self.enviroment,
                           min_trading_unit = min_trading_unit,
                           max_trading_unit= max_trading_unit,
                           delayed_reward_threshold = delayed_reward_threshold)

        #학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        #벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data_shape[1]

        #신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        #가시화 모듈
        self.visualizer = Visualizer()

        #메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        #에포크 관련 정보
        self.loss = 0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        #로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network=None,
                           activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)


    def init_policy_network(self, shared_network=None,
            activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.enviroment.observation


