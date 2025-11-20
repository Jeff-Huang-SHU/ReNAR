"""JAX implementation of CLRS basic network."""

import functools

from typing import Dict, List, Optional, Tuple

import chex

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@chex.dataclass
class _MessagePassingScanState:
  hint_preds: chex.Array
  output_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]
  mlm_hint_preds: List[chex.Array]
  mlm_truths: List[Dict[str, _Array]]
  mlm_masks: List[Dict[str,_Array]]


@chex.dataclass
class _MessagePassingOutputChunked:
  hint_preds: chex.Array
  output_preds: chex.Array
  mlm_hint_preds: List[chex.Array]
  mlm_truths: List[Dict[str, _Array]]
  mlm_masks:List[Dict[str,_Array]]

@chex.dataclass
class MessagePassingStateChunked:
  inputs: chex.Array
  hints: chex.Array
  is_first: chex.Array
  hint_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]
  mlm_hint_preds: List[chex.Array]
  mlm_truths: List[Dict[str, _Array]]
  mlm_masks: List[Dict[str,_Array]]

class GateNet(hk.Module):
    def __init__(self, name=None, hidden_dim=128):
        super().__init__(name=name)
        self.o1 = hk.Linear(hidden_dim)
        self.gate1 = hk.Linear(hidden_dim)
        self.gate2 = hk.Linear(hidden_dim)
        self.gate3 = hk.Linear(hidden_dim, b_init=hk.initializers.Constant(-3))

    def __call__(self, nxt_hidden, msg_node):
        z = self.o1(nxt_hidden)
        ret = z + msg_node
        gate = jax.nn.sigmoid(self.gate3(jax.nn.relu(self.gate1(nxt_hidden) + self.gate2(msg_node))))
        ret = ret * gate + nxt_hidden * (1 - gate)
        return ret

class Net(hk.Module):
  """Building blocks (networks) used to encode and decode messages."""

  def __init__(
      self,
      spec: List[_Spec],
      hidden_dim: int,
      encode_hints: bool,
      decode_hints: bool,
      processor_factory: processors.ProcessorFactory,
      use_lstm: bool,
      encoder_init: str,
      dropout_prob: float,
      hint_teacher_forcing: float,
      hint_repred_mode='soft',
      nb_dims=None,
      nb_msg_passing_steps=1,
      debug=False,
      mlm_processor_factory: processors.ProcessorFactory=None,
      mlm_ratio:float=0.15,
      max_K_times:int=1,
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    super().__init__(name=name)

    self._dropout_prob = dropout_prob
    self._hint_teacher_forcing = hint_teacher_forcing
    self._hint_repred_mode = hint_repred_mode
    self.spec = spec
    self.hidden_dim = hidden_dim
    self.encode_hints = encode_hints
    self.decode_hints = decode_hints
    self.processor_factory = processor_factory
    self.mlm_processor_factory = mlm_processor_factory
    self.nb_dims = nb_dims
    self.use_lstm = use_lstm
    self.encoder_init = encoder_init
    self.nb_msg_passing_steps = nb_msg_passing_steps
    self.debug = debug
    self.mlm_ratio = mlm_ratio
    self.k_times = min(int(1 / self.mlm_ratio), max_K_times) if self.mlm_ratio > 0 else 1 
    self.replace_rate = 0.15

  def create_mask_token(self, 
                      features_list: List[_Features]):
    """ create mask tokens for MLM training """
    algo_mask_tokens = []
    for features in features_list:
      mask_tokens = {}
      inputs = features.inputs 
      hints = features.hints   
      for traj in [inputs, hints]:
        for dp in traj:
          name, type, loc = dp.name, dp.type_, dp.location
          stage = 'input' if traj in inputs else 'hint'
          in_dim = dp.data.shape[-1] if type == _Type.CATEGORICAL else 1
          mask_tokens[name] = hk.get_parameter(
                                    f"{name}_{type}_{loc}_{stage}_MLM_token",
                                    shape=(1,in_dim), 
                                    dtype=jnp.float32,
                                    init=hk.initializers.TruncatedNormal() 
                                  )
      algo_mask_tokens.append(mask_tokens)
    return algo_mask_tokens
  
  def _msg_passing_step(self,
                        mp_state: _MessagePassingScanState,
                        i: int,
                        hints: List[_DataPoint],
                        repred: bool,
                        lengths: chex.Array,
                        batch_size: int,
                        nb_nodes: int,
                        inputs: _Trajectory,
                        first_step: bool,
                        spec: _Spec,
                        encs: Dict[str, List[hk.Module]],
                        decs: Dict[str, Tuple[hk.Module]],
                        return_hints: bool,
                        return_all_outputs: bool,
                        mlm_decs: Dict[str, Tuple[hk.Module]],
                        mlm_tokens,
                        gatenet_list
                        ):
    if self.decode_hints and not first_step:
      assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
      hard_postprocess = (self._hint_repred_mode == 'hard' or
                          (self._hint_repred_mode == 'hard_on_eval' and repred))
      decoded_hint = decoders.postprocess(spec,
                                          mp_state.hint_preds,
                                          sinkhorn_temperature=0.1,
                                          sinkhorn_steps=25,
                                          hard=hard_postprocess)
    if repred and self.decode_hints and not first_step:
      cur_hint = []
      for hint in decoded_hint:
        cur_hint.append(decoded_hint[hint])
    else:
      cur_hint = []
      needs_noise = (self.decode_hints and not first_step and
                     self._hint_teacher_forcing < 1.0)
      if needs_noise:
        force_mask = jax.random.bernoulli(
            hk.next_rng_key(), self._hint_teacher_forcing,
            (batch_size,))
      else:
        force_mask = None
      for hint in hints:
        hint_data = jnp.asarray(hint.data)[i]
        _, loc, typ = spec[hint.name]
        if needs_noise:
          if (typ == _Type.POINTER and
              decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
            hint_data = hk.one_hot(hint_data, nb_nodes)
            typ = _Type.SOFT_POINTER
          hint_data = jnp.where(_expand_to(force_mask, hint_data),
                                hint_data,
                                decoded_hint[hint.name].data)
        cur_hint.append(
            probing.DataPoint(
                name=hint.name, location=loc, type_=typ, data=hint_data))
    hiddens, output_preds_cand, hint_preds, lstm_state, mlm_hint_preds, mlm_truths, mlm_masks = self._one_step_pred(
        inputs, cur_hint, mp_state.hiddens,
        batch_size, nb_nodes, mp_state.lstm_state,
        spec, encs, decs, repred, mlm_decs, mlm_tokens, gatenet_list)

    if first_step:
      output_preds = output_preds_cand
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = _is_not_done_broadcast(lengths, i,
                                             output_preds_cand[outp])
        output_preds[outp] = is_not_done * output_preds_cand[outp] + (
            1.0 - is_not_done) * mp_state.output_preds[outp]

    new_mp_state = _MessagePassingScanState(  
        hint_preds=hint_preds,
        output_preds=output_preds,
        hiddens=hiddens,
        lstm_state=lstm_state,
        mlm_hint_preds=mlm_hint_preds,
        mlm_truths=mlm_truths,
        mlm_masks=mlm_masks)
    
    accum_mp_state = _MessagePassingScanState(  
        hint_preds=hint_preds if return_hints else None,
        output_preds=output_preds if return_all_outputs else None,
        hiddens=hiddens if self.debug else None, lstm_state=None,
        mlm_hint_preds = mlm_hint_preds if return_hints else None,
        mlm_truths=mlm_truths if return_hints else None,
        mlm_masks=mlm_masks if return_hints else None,)

    return new_mp_state, accum_mp_state

  def __call__(self, features_list: List[_Features], repred: bool,
               algorithm_index: int,
               return_hints: bool,
               return_all_outputs: bool):
    """Process one batch of data.

    Args:
      features_list: A list of _Features objects, each with the inputs, hints
        and lengths for a batch o data corresponding to one algorithm.
        The list should have either length 1, at train/evaluation time,
        or length equal to the number of algorithms this Net is meant to
        process, at initialization.
      repred: False during training, when we have access to ground-truth hints.
        True in validation/test mode, when we have to use our own
        hint predictions.
      algorithm_index: Which algorithm is being processed. It can be -1 at
        initialisation (either because we are initialising the parameters of
        the module or because we are intialising the message-passing state),
        meaning that all algorithms should be processed, in which case
        `features_list` should have length equal to the number of specs of
        the Net. Otherwise, `algorithm_index` should be
        between 0 and `length(self.spec) - 1`, meaning only one of the
        algorithms will be processed, and `features_list` should have length 1.
      return_hints: Whether to accumulate and return the predicted hints,
        when they are decoded.
      return_all_outputs: Whether to return the full sequence of outputs, or
        just the last step's output.

    Returns:
      A 2-tuple with (output predictions, hint predictions)
      for the selected algorithm.
    """
    if algorithm_index == -1:
      logging.info(f"mlm_ratio: {self.mlm_ratio}, k_times: {self.k_times}, replace_rate: {self.replace_rate}")
      algorithm_indices = range(len(features_list))
    else:
      algorithm_indices = [algorithm_index]
    assert len(algorithm_indices) == len(features_list)

    self.encoders, self.decoders = self._construct_encoders_decoders(tag='CLRS')
    _, self.mlm_decoders = self._construct_encoders_decoders(tag='mlm')
    self.mlm_mask_tokens = self.create_mask_token(features_list)

    self.processor = self.processor_factory(self.hidden_dim)
    self.mlm_processor = self.mlm_processor_factory(self.hidden_dim)
    gatenet_list = [
      GateNet(name='node_gatenet', hidden_dim=self.hidden_dim),
      GateNet(name='edge_gatenet', hidden_dim=self.hidden_dim),
    ]
    if self.use_lstm:
      self.lstm = hk.LSTM(
          hidden_size=self.hidden_dim,
          name='processor_lstm')
      lstm_init = self.lstm.initial_state
    else:
      self.lstm = None
      lstm_init = lambda x: 0

    for algorithm_index, features in zip(algorithm_indices, features_list):
      inputs = features.inputs
      hints = features.hints
      lengths = features.lengths

      batch_size, nb_nodes = _data_dimensions(features)
      nb_mp_steps = max(1, hints[0].data.shape[0] - 1) 
      hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

      if self.use_lstm:
        lstm_state = lstm_init(batch_size * nb_nodes)
        lstm_state = jax.tree_util.tree_map(
            lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
            lstm_state)
      else:
        lstm_state = None

      mp_state = _MessagePassingScanState(  
          hint_preds=None, output_preds=None,
          hiddens=hiddens, lstm_state=lstm_state,
          mlm_hint_preds = None,
          mlm_truths = None,
          mlm_masks = None)

      common_args = dict(
          hints=hints,
          repred=repred,
          inputs=inputs,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
          lengths=lengths,
          spec=self.spec[algorithm_index],
          encs=self.encoders[algorithm_index],
          decs=self.decoders[algorithm_index],
          return_hints=return_hints,
          return_all_outputs=return_all_outputs,
          mlm_decs = self.mlm_decoders[algorithm_index],
          mlm_tokens = self.mlm_mask_tokens[algorithm_index],
          gatenet_list= gatenet_list
          )
      mp_state, lean_mp_state = self._msg_passing_step(
          mp_state,
          i=0,
          first_step=True,
          **common_args)

      scan_fn = functools.partial(
          self._msg_passing_step,
          first_step=False,
          **common_args)

      output_mp_state, accum_mp_state = hk.scan(
          scan_fn,
          mp_state,
          jnp.arange(nb_mp_steps - 1) + 1,
          length=nb_mp_steps - 1)

    accum_mp_state = jax.tree_util.tree_map(
        lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
        lean_mp_state, accum_mp_state)

    def invert(d):
      """Dict of lists -> list of dicts."""
      if d:
        return [dict(zip(d, i)) for i in zip(*d.values())]

    if return_all_outputs:
      output_preds = {k: jnp.stack(v)
                      for k, v in accum_mp_state.output_preds.items()}
    else:
      output_preds = output_mp_state.output_preds
    hint_preds = invert(accum_mp_state.hint_preds)

    mlm_hint_preds = [invert(accum_mp_state.mlm_hint_preds[idx]) if accum_mp_state.mlm_hint_preds else None for idx in range(self.k_times)]
    mlm_truths = [invert(accum_mp_state.mlm_truths[idx]) if accum_mp_state.mlm_truths else None for idx in range(self.k_times)]
    mlm_masks = [invert(accum_mp_state.mlm_masks[idx]) if accum_mp_state.mlm_masks else None for idx in range(self.k_times)]

    if self.debug:
      hiddens = jnp.stack([v for v in accum_mp_state.hiddens])
      return output_preds, hint_preds, hiddens, mlm_hint_preds, mlm_truths, mlm_masks

    return output_preds, hint_preds, mlm_hint_preds, mlm_truths, mlm_masks

  def _construct_encoders_decoders(self,tag:str):
    """Constructs encoders and decoders, separate for each algorithm."""
    encoders_ = []
    decoders_ = []
    enc_algo_idx = None
    for (algo_idx, spec) in enumerate(self.spec):
      enc = {}
      dec = {}
      for name, (stage, loc, t) in spec.items():
        if stage == _Stage.INPUT or (
            stage == _Stage.HINT and self.encode_hints):
          
          if name == specs.ALGO_IDX_INPUT_NAME:
            if enc_algo_idx is None:
              enc_algo_idx = [hk.Linear(self.hidden_dim,
                                        name=f'{tag}_{name}_enc_linear')]
            enc[name] = enc_algo_idx
          else:
            enc[name] = encoders.construct_encoders(
                stage, loc, t, hidden_dim=self.hidden_dim,
                init=self.encoder_init,
                name=f'{tag}_algo_{algo_idx}_{name}')

        if stage == _Stage.OUTPUT or (
            stage == _Stage.HINT and self.decode_hints):
          
          dec[name] = decoders.construct_decoders(
              loc, t, hidden_dim=self.hidden_dim,
              nb_dims=self.nb_dims[algo_idx][name],
              name=f'{tag}_algo_{algo_idx}_{name}')
      encoders_.append(enc)
      decoders_.append(dec)

    return encoders_, decoders_
  def random_mask(self,
                  dp:_DataPoint,
                  mlm_token:hk.Module,
                  method:str,
                  mask_stage_domain: List[_Location],
                  ramdom_node_mask: _Array=None
    ):
    """Generates Masked Data and Masks."""
    original_data = dp.data
    if method == 'method1':
      """ Fully Random Mask. This is the method we used in this paper.
          1. With probability `mlm_ratio`, each feature value of a node is independently set to zero.  
      """
      if dp.location in mask_stage_domain: 
        mask = jax.random.bernoulli(hk.next_rng_key(), self.mlm_ratio, original_data.shape)
      else:
        mask = jax.random.bernoulli(hk.next_rng_key(), 0.0, original_data.shape)
      mlm_data = jnp.where(mask, jnp.zeros_like(original_data), original_data)
    elif method == 'method2':
      if dp.location in mask_stage_domain: 
        assert dp.location != _Location.GRAPH
        batch_size, nb_nodes = original_data.shape[0], original_data.shape[1]
        if dp.type_ != _Type.CATEGORICAL:
          original_data = jnp.expand_dims(original_data, axis=-1)
        while len(mlm_token.shape) < len(original_data.shape):
          mlm_token = jnp.expand_dims(mlm_token, axis=0)
        
        mask_rng, replace_rng, shuffle_rng = hk.next_rng_key(), hk.next_rng_key(), hk.next_rng_key()

        mask = jax.random.bernoulli(mask_rng, self.mlm_ratio, original_data.shape) 
        replace_mask = jax.random.bernoulli(replace_rng, self.replace_rate, original_data.shape) 

        mlm_data = jnp.where(
            jnp.logical_and(mask, jnp.logical_not(replace_mask)),
            mlm_token,
            original_data
        )
        
        mlm_data = jnp.where(
            jnp.logical_and(mask, replace_mask),
            jnp.zeros_like(mlm_data),
            mlm_data
        )
        random_indices = jax.random.randint(shuffle_rng, (batch_size, nb_nodes), 0, nb_nodes)
        random_node_indices_for_replacement = jax.random.randint(shuffle_rng, (batch_size, nb_nodes), 0, nb_nodes)
        def select_random_features_for_batch(batch_data, random_indices):
            return batch_data[random_indices]

        random_features_for_replacement = jax.vmap(select_random_features_for_batch)(original_data, random_node_indices_for_replacement)

        mlm_data = jnp.where(
            jnp.logical_and(mask, replace_mask),
            random_features_for_replacement,
            mlm_data
        )
        if dp.type_ != _Type.CATEGORICAL:
          mlm_data = jnp.squeeze(mlm_data, axis=-1)
          mask = jnp.squeeze(mask, axis=-1)
      else:
        mask = jax.random.bernoulli(hk.next_rng_key(), 0.0, original_data.shape) 
        mlm_data = original_data
    elif method == 'method3':

      if ramdom_node_mask is None:
        raise ValueError('Nede Node MASK!, Shape to be (B,N)')

      if dp.location in mask_stage_domain: 
          if dp.type_ != _Type.CATEGORICAL:
            original_data = jnp.expand_dims(original_data, axis=-1)
          while len(mlm_token.shape) < len(original_data.shape):
            mlm_token = jnp.expand_dims(mlm_token, axis=0)

          mask = ramdom_node_mask
          while len(mask.shape) < len(original_data.shape):
            mask = jnp.expand_dims(mask, axis=-1)

          replace_rng, shuffle_rng = hk.next_rng_key(), hk.next_rng_key()
  
          
          replace_mask = jax.random.bernoulli(replace_rng, self.replace_rate, original_data.shape) 
          
          mlm_data = jnp.where(
              jnp.logical_and(mask, jnp.logical_not(replace_mask)),
              mlm_token,
              original_data
          )
          
          mlm_data = jnp.where(
              jnp.logical_and(mask, replace_mask),
              jnp.zeros_like(mlm_data),
              mlm_data
          )
          mlm_data = jnp.squeeze(mlm_data, axis=-1)
          mask = jnp.squeeze(mask, axis=-1)
      else:
          mask = jax.random.bernoulli(hk.next_rng_key(), 0.0, original_data.shape) 
          mlm_data = original_data
    elif method == 'method4':


      if dp.location in mask_stage_domain: 
        assert dp.location != _Location.GRAPH
        
        if dp.type_ == _Type.SCALAR: 
          batch_size, nb_nodes = original_data.shape[0], original_data.shape[1]
          original_data = jnp.expand_dims(original_data, axis=-1)
          while len(mlm_token.shape) < len(original_data.shape):
            mlm_token = jnp.expand_dims(mlm_token, axis=0)
          
          mask_rng, replace_rng, shuffle_rng = hk.next_rng_key(), hk.next_rng_key(), hk.next_rng_key()
          mask = jax.random.bernoulli(mask_rng, self.mlm_ratio, original_data.shape) 
          replace_mask = jax.random.bernoulli(replace_rng, self.replace_rate, original_data.shape) 
          mlm_data = jnp.where(
              jnp.logical_and(mask, jnp.logical_not(replace_mask)),
              mlm_token,
              original_data
          )
          random_indices = jax.random.randint(shuffle_rng, (batch_size, nb_nodes), 0, nb_nodes)
          random_node_indices_for_replacement = jax.random.randint(shuffle_rng, (batch_size, nb_nodes), 0, nb_nodes)

          def select_random_features_for_batch(batch_data, random_indices):
              return batch_data[random_indices]

          random_features_for_replacement = jax.vmap(select_random_features_for_batch)(original_data, random_node_indices_for_replacement)
          mlm_data = jnp.where(
              jnp.logical_and(mask, replace_mask),
              random_features_for_replacement,
              mlm_data
          )
          mlm_data = jnp.squeeze(mlm_data, axis=-1)
          mask = jnp.squeeze(mask, axis=-1)
          
        else:
          zero_replace_rate = 0.5
          mask_rng, replace_rng = hk.next_rng_key(), hk.next_rng_key()
          
          mask = jax.random.bernoulli(mask_rng, self.mlm_ratio, original_data.shape)
          replace_mask = jax.random.bernoulli(replace_rng, zero_replace_rate, original_data.shape) 
          mlm_data = jnp.where(
              jnp.logical_and(mask, jnp.logical_not(replace_mask)),
              jnp.zeros_like(original_data),
              original_data
          )
          mlm_data = jnp.where(
              jnp.logical_and(mask, replace_mask),
              jnp.ones_like(mlm_data),
              mlm_data
          )
      else:
        mask = jax.random.bernoulli(hk.next_rng_key(), 0.0, original_data.shape) 
        mlm_data = original_data
    else:
      raise ValueError('Not in methods domain!')
    mlm_dp = probing.DataPoint(
                name=dp.name, location=dp.location, type_=dp.type_, data=mlm_data)

    return mlm_dp, mask
  
  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      batch_size: int,
      nb_nodes: int,
      lstm_state: Optional[hk.LSTMState],
      spec: _Spec,
      encs: Dict[str, List[hk.Module]],
      decs: Dict[str, Tuple[hk.Module]],
      repred: bool,
      mlm_decs: Dict[str, Tuple[hk.Module]],
      mlm_tokens,
      gatenet_list
  ):
    """Generates one-step predictions."""
    
    
    k_times = self.k_times
    mlm_node_fts = [jnp.zeros((batch_size, nb_nodes, self.hidden_dim)) for _ in range(k_times)]
    mlm_edge_fts = [jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim)) for _ in range(k_times)]
    mlm_adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

    node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
    
    
    mlm_truths, mlm_masks = [{name:None for name in encs} for _ in range(k_times)], [{name:None for name in encs} for _ in range(k_times)]
    # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode node/edge/graph features from inputs and (optionally) hints.

    trajectories = [inputs]
    if self.encode_hints:
      trajectories.append(hints)

    for trajectory in trajectories:
      for dp in trajectory:
        try:
          dp = encoders.preprocess(dp, nb_nodes)
          original_data = dp.data
          assert dp.type_ != _Type.SOFT_POINTER
          encoder = encs[dp.name]
          for idx in range(k_times):
            if not repred: 
              mlm_dp, mask = self.random_mask(dp, 
                                              mlm_tokens[dp.name], 
                                              'method1', 
                                              [_Location.NODE],
                                              
                                              ramdom_node_mask=jax.random.bernoulli(hk.next_rng_key(), self.mlm_ratio, original_data.shape[:2]))
              mlm_truths[idx][mlm_dp.name], mlm_masks[idx][mlm_dp.name] = original_data, mask
            else:
              mlm_dp = dp
            mlm_node_fts[idx] = encoders.accum_node_fts(encoder, mlm_dp, mlm_node_fts[idx])
            mlm_edge_fts[idx] = encoders.accum_edge_fts(encoder, mlm_dp, mlm_edge_fts[idx])
        
          mlm_adj_mat = encoders.accum_adj_mat(mlm_dp, mlm_adj_mat)
          adj_mat = encoders.accum_adj_mat(dp, adj_mat)
          edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
          node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
          graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)

        except Exception as e:
          raise Exception(f'Failed to process {dp}') from e
    # MLM PROCESS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mlm_node_hidden = [hidden for _ in range(k_times)]
    mlm_edge_hidden = [None for _ in range(k_times)]
    mlm_h_t, mlm_e_t = [None for _ in range(k_times)], [None for _ in range(k_times)]
    for idx in range(k_times):
      for _ in range(self.nb_msg_passing_steps): # M layers
        mlm_node_hidden[idx], mlm_edge_hidden[idx] = self.mlm_processor(
            mlm_node_fts[idx],
            mlm_edge_fts[idx],
            graph_fts,
            mlm_adj_mat,
            mlm_node_hidden[idx],
            batch_size=batch_size,
            nb_nodes=nb_nodes,
        )
      mlm_h_t[idx] = jnp.concatenate([mlm_node_fts[idx], mlm_node_hidden[idx]], axis=-1)
      if mlm_edge_hidden is not None:
        mlm_e_t[idx] = jnp.concatenate([mlm_edge_fts[idx], mlm_edge_hidden[idx]], axis=-1)
      else:
        mlm_e_t[idx] = mlm_edge_fts[idx]
    # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nxt_hidden = hidden
    node_input_for_processor = gatenet_list[0](node_fts, jnp.mean(jnp.stack(mlm_node_hidden),axis=0))
    edge_input_for_processor = gatenet_list[1](edge_fts, jnp.mean(jnp.stack(mlm_edge_hidden),axis=0)) if mlm_edge_hidden is not None else edge_fts
    for _ in range(self.nb_msg_passing_steps):
      nxt_hidden, nxt_edge = self.processor(
          node_input_for_processor,
          edge_input_for_processor,
          graph_fts,
          mlm_adj_mat, 
          nxt_hidden,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
      )

    if not repred:      # dropout only on training
      nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

    if self.use_lstm:
      # lstm doesn't accept multiple batch dimensions (in our case, batch and
      # nodes), so we vmap over the (first) batch dimension.
      nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
    else:
      nxt_lstm_state = None

    h_t = jnp.concatenate([node_input_for_processor, hidden, nxt_hidden], axis=-1)
    if nxt_edge is not None:
      e_t = jnp.concatenate([edge_input_for_processor, nxt_edge], axis=-1)
    else:
      e_t = edge_input_for_processor

    # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decode features and (optionally) hints.
    hint_preds, output_preds = decoders.decode_fts(
        decoders=decs,
        spec=spec,
        h_t=h_t,
        adj_mat=mlm_adj_mat,
        edge_fts=e_t,
        graph_fts=graph_fts,
        inf_bias=self.processor.inf_bias,
        inf_bias_edge=self.processor.inf_bias_edge,
        repred=repred,
    )
    mlm_hint_preds = []
    for idx in range(k_times):
      mlm_hint_pred, _ = decoders.decode_fts(
          decoders=mlm_decs,
          spec=spec,
          h_t=mlm_h_t[idx],
          adj_mat=mlm_adj_mat,
          edge_fts=mlm_e_t[idx],
          graph_fts=graph_fts,
          inf_bias=self.mlm_processor.inf_bias,
          inf_bias_edge=self.mlm_processor.inf_bias_edge,
          repred=repred,
      )
      mlm_hint_preds.append(mlm_hint_pred)
    
    return nxt_hidden, output_preds, hint_preds, nxt_lstm_state, mlm_hint_preds, mlm_truths, mlm_masks


class NetChunked(Net):
  """A Net that will process time-chunked data instead of full samples."""

  def _msg_passing_step(self,
                        mp_state: MessagePassingStateChunked,
                        xs,
                        repred: bool,
                        init_mp_state: bool,
                        batch_size: int,
                        nb_nodes: int,
                        spec: _Spec,
                        encs: Dict[str, List[hk.Module]],
                        decs: Dict[str, Tuple[hk.Module]],
                        mlm_decs: Dict[str, Tuple[hk.Module]],
                        mlm_ratio:float=0.15
                        ):
    """Perform one message passing step.

    This function is unrolled along the time axis to process a data chunk.

    Args:
      mp_state: message-passing state. Includes the inputs, hints,
        beginning-of-sample markers, hint predictions, hidden and lstm state
        to be used for prediction in the current step.
      xs: A 3-tuple of with the next timestep's inputs, hints, and
        beginning-of-sample markers. These will replace the contents of
        the `mp_state` at the output, in readiness for the next unroll step of
        the chunk (or the first step of the next chunk). Besides, the next
        timestep's hints are necessary to compute diffs when `decode_diffs`
        is True.
      repred: False during training, when we have access to ground-truth hints.
        True in validation/test mode, when we have to use our own
        hint predictions.
      init_mp_state: Indicates if we are calling the method just to initialise
        the message-passing state, before the beginning of training or
        validation.
      batch_size: Size of batch dimension.
      nb_nodes: Number of nodes in graph.
      spec: The spec of the algorithm being processed.
      encs: encoders for the algorithm being processed.
      decs: decoders for the algorithm being processed.
    Returns:
      A 2-tuple with the next mp_state and an output consisting of
      hint predictions and output predictions.
    """
    def _as_prediction_data(hint):
      if hint.type_ == _Type.POINTER:
        return hk.one_hot(hint.data, nb_nodes)
      return hint.data

    nxt_inputs, nxt_hints, nxt_is_first = xs
    inputs = mp_state.inputs
    is_first = mp_state.is_first
    hints = mp_state.hints
    if init_mp_state:
      prev_hint_preds = {h.name: _as_prediction_data(h) for h in hints}
      hints_for_pred = hints
    else:
      prev_hint_preds = mp_state.hint_preds
      if self.decode_hints:
        if repred:
          force_mask = jnp.zeros(batch_size, dtype=bool)
        elif self._hint_teacher_forcing == 1.0:
          force_mask = jnp.ones(batch_size, dtype=bool)
        else:
          force_mask = jax.random.bernoulli(
              hk.next_rng_key(), self._hint_teacher_forcing,
              (batch_size,))
        assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
        hard_postprocess = (
            self._hint_repred_mode == 'hard' or
            (self._hint_repred_mode == 'hard_on_eval' and repred))
        decoded_hints = decoders.postprocess(spec,
                                             prev_hint_preds,
                                             sinkhorn_temperature=0.1,
                                             sinkhorn_steps=25,
                                             hard=hard_postprocess)
        hints_for_pred = []
        for h in hints:
          typ = h.type_
          hint_data = h.data
          if (typ == _Type.POINTER and
              decoded_hints[h.name].type_ == _Type.SOFT_POINTER):
            hint_data = hk.one_hot(hint_data, nb_nodes)
            typ = _Type.SOFT_POINTER
          hints_for_pred.append(probing.DataPoint(
              name=h.name, location=h.location, type_=typ,
              data=jnp.where(_expand_to(is_first | force_mask, hint_data),
                             hint_data, decoded_hints[h.name].data)))
      else:
        hints_for_pred = hints

    hiddens = jnp.where(is_first[..., None, None], 0.0, mp_state.hiddens)
    if self.use_lstm:
      lstm_state = jax.tree_util.tree_map(
          lambda x: jnp.where(is_first[..., None, None], 0.0, x),
          mp_state.lstm_state)
    else:
      lstm_state = None
    hiddens, output_preds, hint_preds, lstm_state, mlm_hint_preds, mlm_truths, mlm_masks = self._one_step_pred(
        inputs, hints_for_pred, hiddens,
        batch_size, nb_nodes, lstm_state,
        spec, encs, decs, repred, mlm_decs, mlm_ratio)

    new_mp_state = MessagePassingStateChunked(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hiddens=hiddens, lstm_state=lstm_state, hint_preds=hint_preds,
        inputs=nxt_inputs, hints=nxt_hints, is_first=nxt_is_first,
        mlm_hint_preds=mlm_hint_preds, mlm_truths=mlm_truths, mlm_masks=mlm_masks )
    
    mp_output = _MessagePassingOutputChunked(  # pytype: disable=wrong-arg-types  # numpy-scalars
        hint_preds=hint_preds,
        output_preds=output_preds,
        mlm_hint_preds=mlm_hint_preds, mlm_truths=mlm_truths, mlm_masks=mlm_masks)
    
    return new_mp_state, mp_output

  def __call__(self, features_list: List[_FeaturesChunked],
               mp_state_list: List[MessagePassingStateChunked],
               repred: bool, init_mp_state: bool,
               algorithm_index: int):
    """Process one chunk of data.

    Args:
      features_list: A list of _FeaturesChunked objects, each with the
        inputs, hints and beginning- and end-of-sample markers for
        a chunk (i.e., fixed time length) of data corresponding to one
        algorithm. All features are expected
        to have dimensions chunk_length x batch_size x ...
        The list should have either length 1, at train/evaluation time,
        or length equal to the number of algorithms this Net is meant to
        process, at initialization.
      mp_state_list: list of message-passing states. Each message-passing state
        includes the inputs, hints, beginning-of-sample markers,
        hint prediction, hidden and lstm state from the end of the previous
        chunk, for one algorithm. The length of the list should be the same
        as the length of `features_list`.
      repred: False during training, when we have access to ground-truth hints.
        True in validation/test mode, when we have to use our own hint
        predictions.
      init_mp_state: Indicates if we are calling the network just to initialise
        the message-passing state, before the beginning of training or
        validation. If True, `algorithm_index` (see below) must be -1 in order
        to initialize the message-passing state of all algorithms.
      algorithm_index: Which algorithm is being processed. It can be -1 at
        initialisation (either because we are initialising the parameters of
        the module or because we are intialising the message-passing state),
        meaning that all algorithms should be processed, in which case
        `features_list` and `mp_state_list` should have length equal to the
        number of specs of the Net. Otherwise, `algorithm_index` should be
        between 0 and `length(self.spec) - 1`, meaning only one of the
        algorithms will be processed, and `features_list` and `mp_state_list`
        should have length 1.

    Returns:
      A 2-tuple consisting of:
      - A 2-tuple with (output predictions, hint predictions)
        for the selected algorithm. Each of these has
        chunk_length x batch_size x ... data, where the first time
        slice contains outputs for the mp_state
        that was passed as input, and the last time slice contains outputs
        for the next-to-last slice of the input features. The outputs that
        correspond to the final time slice of the input features will be
        calculated when the next chunk is processed, using the data in the
        mp_state returned here (see below). If `init_mp_state` is True,
        we return None instead of the 2-tuple.
      - The mp_state (message-passing state) for the next chunk of data
        of the selected algorithm. If `init_mp_state` is True, we return
        initial mp states for all the algorithms.
    """
    if algorithm_index == -1:
      algorithm_indices = range(len(features_list))
    else:
      algorithm_indices = [algorithm_index]
      assert not init_mp_state  # init state only allowed with all algorithms
    assert len(algorithm_indices) == len(features_list)
    assert len(algorithm_indices) == len(mp_state_list)

    self.encoders, self.decoders = self._construct_encoders_decoders(tag='CLRS')
    _, self.mlm_decoders = self._construct_encoders_decoders(tag='mlm')

    self.processor = self.processor_factory(self.hidden_dim)
    self.mlm_processor = self.mlm_processor_factory(self.hidden_dim)
    # Optionally construct LSTM.
    if self.use_lstm:
      self.lstm = hk.LSTM(
          hidden_size=self.hidden_dim,
          name='processor_lstm')
      lstm_init = self.lstm.initial_state
    else:
      self.lstm = None
      lstm_init = lambda x: 0

    if init_mp_state:
      output_mp_states = []
      for algorithm_index, features, mp_state in zip(
          algorithm_indices, features_list, mp_state_list):
        inputs = features.inputs
        hints = features.hints
        batch_size, nb_nodes = _data_dimensions_chunked(features)

        if self.use_lstm:
          lstm_state = lstm_init(batch_size * nb_nodes)
          lstm_state = jax.tree_util.tree_map(
              lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
              lstm_state)
          mp_state.lstm_state = lstm_state
        mp_state.inputs = jax.tree_util.tree_map(lambda x: x[0], inputs)
        mp_state.hints = jax.tree_util.tree_map(lambda x: x[0], hints)
        mp_state.is_first = jnp.zeros(batch_size, dtype=int)
        mp_state.hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        next_is_first = jnp.ones(batch_size, dtype=int)

        mp_state, _ = self._msg_passing_step(
            mp_state,
            (mp_state.inputs, mp_state.hints, next_is_first),
            repred=repred,
            init_mp_state=True,
            batch_size=batch_size,
            nb_nodes=nb_nodes,
            spec=self.spec[algorithm_index],
            encs=self.encoders[algorithm_index],
            decs=self.decoders[algorithm_index],
            mlm_decs =  self.mlm_decoders[algorithm_index],
            mlm_ratio= self.mlm_ratio
            )
        output_mp_states.append(mp_state)
      return None, output_mp_states

    for algorithm_index, features, mp_state in zip(
        algorithm_indices, features_list, mp_state_list):
      inputs = features.inputs
      hints = features.hints
      is_first = features.is_first
      batch_size, nb_nodes = _data_dimensions_chunked(features)

      scan_fn = functools.partial(
          self._msg_passing_step,
          repred=repred,
          init_mp_state=False,
          batch_size=batch_size,
          nb_nodes=nb_nodes,
          spec=self.spec[algorithm_index],
          encs=self.encoders[algorithm_index],
          decs=self.decoders[algorithm_index],
          mlm_decs =  self.mlm_decoders[algorithm_index],
          mlm_ratio= self.mlm_ratio
          )

      mp_state, scan_output = hk.scan(
          scan_fn,
          mp_state,
          (inputs, hints, is_first),
      )

    # We only return the last algorithm's output and state. That's because
    # the output only matters when a single algorithm is processed; the case
    # `algorithm_index==-1` (meaning all algorithms should be processed)
    # is used only to init parameters.
    return (scan_output.output_preds, scan_output.hint_preds, 
            scan_output.mlm_hint_preds, scan_output.mlm_truths, scan_output.mlm_masks), mp_state


def _data_dimensions(features: _Features) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[:2]
  assert False


def _data_dimensions_chunked(features: _FeaturesChunked) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[1:3]
  assert False


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = jnp.expand_dims(x, -1)
  return x


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):  # pytype: disable=attribute-error  # numpy-scalars
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
