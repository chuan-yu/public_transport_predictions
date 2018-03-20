import os

class LSTMConfig():
    def __init__(self, station_code):
        self.train_batch_size = 32
        self.state_size = [200, 200]
        self.feature_len = 3
        self.output_time_steps = 10
        self.input_time_steps = 50
        self.lr = 0.001
        self.num_epochs = 100
        self.keep_prob = 1.0
        self.lr_decay = 1.0
        self.checkpoint = os.path.join("checkpoints/mrt-(200, 200)-lr0.005", str(station_code), "checkpoint.ckpt")
        self.write_summary = False
        self.tensorboard_dir = "summaries/lr0.01-decay_0.5_50"

class LSTMConfigBus():
    def __init__(self, station_code):
        self.train_batch_size = 128
        self.state_size = [200, 200, 200]
        self.feature_len = 4
        self.output_time_steps = 10
        self.input_time_steps = 50
        self.lr = 0.005
        self.num_epochs = 400
        self.keep_prob = 1.0
        self.lr_decay = 1.0
        self.checkpoint = os.path.join("../checkpoints/bus/bus-(200, 200, 200)-lr0.005", str(station_code), "checkpoint.ckpt")
        self.write_summary = False
        self.tensorboard_dir = "summaries/lr0.01-decay_0.5_50"

class Seq2SeqConfig():
    def __init__(self, station_code):
        self.train_batch_size = 32
        self.state_size = [200]
        self.feature_len = 3
        self.input_time_steps = 50
        self.output_time_steps = 10
        self.input_size = 3
        self.output_size = 1
        self.lr = 0.0005
        self.num_epochs = 200
        self.keep_prob = 1.0
        self.lr_decay = 1.0
        self.checkpoint = os.path.join("checkpoints/mrt-seq2seq-no_holidays-lr0.001", str(station_code),
                                       "checkpoint.ckpt")
        self.write_summary = False
        self.tensorboard_dir = "summaries/seq2seq_0.01/"


class Seq2SeqConfigBus():
    def __init__(self, station_code):
        self.train_batch_size = 512
        self.state_size = [200]
        self.feature_len = 4
        self.input_time_steps = 50
        self.output_time_steps = 10
        self.input_size = 4
        self.output_size = 1
        self.lr = 0.005
        self.num_epochs = 50
        self.keep_prob = 1.0
        self.lr_decay = 1.0
        self.checkpoint = os.path.join("../checkpoints/bus/bus-seq2seq-no_holidays-lr0.005", str(station_code),
                                       "checkpoint.ckpt")
        self.write_summary = False
        self.tensorboard_dir = "summaries/seq2seq_0.01/"