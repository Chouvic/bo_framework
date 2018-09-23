from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import GPyOpt
import matplotlib
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
from utils.plot import plot_lists, plot_single_list, plot_single_list_same, plot_scatter_xy
matplotlib.pyplot.switch_backend('agg')
import utils.pennflags as flag
from utils import reader
from utils import util
import os
from tensorflow.python.client import device_lib
from utils.pennconfig import get_param_domain, get_kernel, get_model_type


logging = tf.logging
currentDT = datetime.datetime.now()

FLAGS = flag.args
print(FLAGS)

if(FLAGS.v100):
    print('Using v100')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


pngpath = FLAGS.pngpath
batchfilename = FLAGS.batch_filename
cur_param_name = FLAGS.tune_parameter
pngpath = FLAGS.pngpath
curfile = str(FLAGS.bo_epoch)+"_epoba_"
cur_kernel_name = "(" + FLAGS.kernel + ")"

nowtime = str(np.datetime64('now'))
savepath = "results/" + curfile
validsavepath =  savepath+batchfilename + "valid_" + nowtime
trainsavepath = savepath + batchfilename+ "train_" + nowtime
testsavepath = savepath +batchfilename+ "test_" + nowtime
paramssavepath = savepath +batchfilename+ "params"+ nowtime

perplexity_valid_list = []
perplexity_min_test_list = []
perplexity_train_list = []
perplexity_test_list = []
params_list = []

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""


  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob_input < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob_input)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob_update if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob_output < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob_output)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

class SmallConfig(object):
    """GpyOpt config."""
    def __init__(self, params,lr=1, k_input=FLAGS.k_input, k_output=FLAGS.k_output, lr_decay=FLAGS.lr_decay):
        if(FLAGS.run_single == False):
            params = params.flatten()
#        print(params)
        self.learning_rate = lr
        self.keep_prob_input = k_input
        self.keep_prob_output = k_output
        self.lr_decay = lr_decay
        change_num = float(params[0])
        if(FLAGS.tune_parameter == 'learning_rate'):
            self.learning_rate = change_num
        elif(FLAGS.tune_parameter == 'dropout_input'):
            self.keep_prob_input = change_num
        elif(FLAGS.tune_parameter == 'dropout_output'):
            self.keep_prob_output = change_num
        else:
            pass

        # medium configuration
        self.init_scale = FLAGS.init_scale
        self.max_grad_norm = FLAGS.max_grad_norm
        self.num_layers = FLAGS.num_layers
        self.num_steps = FLAGS.num_steps
        self.hidden_size =  FLAGS.hidden_unit
        self.max_epoch = FLAGS.max_epoch
        self.max_max_epoch = FLAGS.max_max_epoch
        self.batch_size = FLAGS.batch_size
        self.vocab_size = FLAGS.vocab_size
        self.rnn_mode = BLOCK
#        print("lr=%.3f keep_input=%.3f keep_output=%.3f lr_decay=%.3f"\
#              %(self.learning_rate, self.keep_prob_input, self.keep_prob_output,\
#                 self.lr_decay))


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)

def get_config(params):
    return SmallConfig(params)


def run(params):
  print(str(params))
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  FLAGS.num_gpus = len(gpus)
#  if FLAGS.num_gpus > len(gpus):
#    raise ValueError(
#        "Your machine has only %d gpus "
#        "which is less than the requested --num_gpus=%d."
#        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config(params)
  eval_config = get_config(params)
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=FLAGS.verbose)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      perplexity_train_list.append(train_perplexity)
      perplexity_valid_list.append(valid_perplexity)
      perplexity_test_list.append(test_perplexity)
      perplexity_min_test_list.append(min(perplexity_test_list))
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
          print("Saving model to %s." % FLAGS.save_path)
          sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
      return valid_perplexity


def run_optimization():
    newkernel = get_kernel(FLAGS.kernel)
    newkernel.lengthscale = FLAGS.length_scale
    domain_temp = get_param_domain(FLAGS.tune_parameter)
    model_type_ = get_model_type(FLAGS.acquisition)
    if(FLAGS.fix_lengthscale):
        newkernel.lengthscale.fix()
    np.random.seed(5)
    lstm_bopt = GPyOpt.methods.BayesianOptimization(run,
                                                    domain=domain_temp,
                                                    model_type=model_type_,
                                                    acquisition_type=FLAGS.acquisition,
                                                    exact_feval=True,
                                                    kernel=newkernel)
    lstm_bopt.run_optimization(FLAGS.bo_epoch)
    ac_png_path = pngpath + batchfilename + FLAGS.acquisition + "acquisition"
    conv_png_path = ac_png_path + "convergence"
    lstm_bopt.plot_acquisition(filename=ac_png_path)
    lstm_bopt.plot_convergence(filename=conv_png_path)

    print("kernel model= "+ str(lstm_bopt.model.get_model_parameters_names()))
    print("------------------------params--------------------------------------")
    for i in range(0,len(perplexity_test_list)):
        print("params: %s  perplexity: %s"%(lstm_bopt.X[i],perplexity_test_list[i]))
        params_list.append(float(lstm_bopt.X[i]))
    print("==Optimum Hyperparams==  Evaluated by valid perplexity")
    print (lstm_bopt.x_opt)
    print("Min Test perplexity %f" % min(perplexity_test_list))
    print("Min Valid perplexity %f" % min(perplexity_valid_list))
    endcurrentDT = datetime.datetime.now()
    print("Training End   time----------- " + str(endcurrentDT))
    print("Training time      ----------- " + str(endcurrentDT-currentDT))
    print("------------------------end bayesian optimisation--------------------------------------")

def run_single(values):
    temp1 = []
    temp1.append(values)
    run(temp1)

def run_grid_search():
    learning_rate = []
    dropout_input = np.linspace(0.1, 0.9, 9)
    dropout_output = np.linspace(0.1, 0.9, 9)
    params_str = ['dropout_input', 'dropout_output',  'learning_rate']
    FLAGS.tune_parameter = params_str[1]

if(FLAGS.run_single == False):
    run_optimization()
else:
#    run_grid_search()
    run_single(FLAGS.run_single_values)

def saveResult(filename, results):
    with open(filename, 'w') as f:
        f.write(str(results))


print("--------------Train perplexity changes-----------------")
print(perplexity_train_list)
print("--------------Valid perplexity changes-----------------")
print(perplexity_valid_list)
print("--------------Test perplexity changes-----------------")
print(perplexity_test_list)

saveResult(trainsavepath, perplexity_train_list)
saveResult(validsavepath, perplexity_valid_list)
saveResult(testsavepath, perplexity_test_list)
saveResult(paramssavepath, params_list)

nowtime = str(np.datetime64('now'))
test_png_path = pngpath+batchfilename+ "test_" + nowtime+cur_param_name
params_png_path = pngpath+batchfilename+ "params_" + nowtime+cur_param_name
cur_best_png_path = pngpath+batchfilename+ "best_cur"+ nowtime+cur_param_name
multi_kernels_path = pngpath+cur_param_name+"multiresults"
scatter_file_path = pngpath +batchfilename+ cur_param_name+"scatter"

plot_single_list(perplexity_test_list,"Test Perplexity",title="Test perplexity performance with epochs"+ cur_kernel_name, \
                 xlabel='Epochs', ylabel='Perplexity', filename=test_png_path)
plot_single_list(params_list,cur_param_name, title=cur_param_name+ " changes with epochs"+ cur_kernel_name, \
                 xlabel='Epochs', ylabel=cur_param_name, filename=params_png_path)
plot_lists(perplexity_test_list, perplexity_min_test_list,l1name='Current_perplexity', l2name='Minimum_perplexity', title='Changes of Current and Best perplexity'+ cur_kernel_name,\
                 xlabel='Epochs', ylabel='Perplexity', filename=cur_best_png_path)
plot_scatter_xy(params_list, perplexity_test_list,title= cur_param_name+" and Test perplexity" + cur_kernel_name ,\
                 xlabel='learning rate', ylabel='Test perplexity', filename=scatter_file_path)
