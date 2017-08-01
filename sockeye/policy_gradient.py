# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet as mx
import numpy as np
from . import bleu

class LogPolicy(mx.operator.CustomOp):
    """
    Computes the gradients w.r.t. sampling based inputs (presumably) from a multinomial distribution.

    https://github.com/dmlc/mxnet/issues/2947

    :param entropy_reg: entropy regularization on the probablity distribution
    :param scale: rescale the gradients
    """
    def __init__(self, entropy_reg, scale):
        super(LogPolicy, self).__init__()
        self.entropy_reg = entropy_reg
        self.scale = scale

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        score = mx.nd.expand_dims(out_grad[0], axis=1).astype('float32')
        action = out_data[0].astype('int32')
        prob = in_data[1]
        is_sample = in_data[2].asnumpy()
        dx = in_grad[1]

        # clipping the policy distribution to avoid the numerical instability when applying log.
        mx.nd.clip(prob, np.finfo(np.float32).eps, 1, out=prob)

        # the gradients of the softmax function w.r.t. inputs
        dx[:] = mx.nd.one_hot(action, prob.shape[1])
        dx[:] /= -prob

        if is_sample[0] > 0:
            # the gradients are weighted by the episode-level returns
            dx[:] = mx.nd.broadcast_mul(dx, score)

            # entropy regularization on the policy distribution
            dx[:] += self.entropy_reg * (1 + mx.nd.log(prob))
        else:
            # if the gradients are computed w.r.t. the true targets, we rescale the gradients.
            # XXX We may keep this until it is proven to be unnecessary.
            #
            # Wu et al. Google's Neural Machine Translation System: Bridging the
            # Gap between Human and Machine Translation, 2016. (\alpha in Eq. 9)

            dx[:] *= self.scale

        # XXX do not propagate the gradients to discrete inputs
        self.assign(in_grad[1], req[1], dx)


@mx.operator.register("LogPolicy")
class LogPolicyProp(mx.operator.CustomOpProp):
    def __init__(self, entropy_reg=0.0, scale=1.0):
        super(LogPolicyProp, self).__init__(need_top_grad=True)

        self.entropy_reg = eval(entropy_reg)
        self.scale = eval(scale)

    def list_arguments(self):
        return ['action', 'prob', 'is_sampled']

    def list_outputs(self):
        return ['action']

    def infer_shape(self, in_shape):
        in_action_shape = in_shape[0]
        out_action_shape = in_shape[0]
        prob_shape = in_shape[1]
        is_sampled_shape = (1,)

        return [in_action_shape, prob_shape, is_sampled_shape], [out_action_shape], []

    def infer_type(self, in_type):
        # TODO check whether type casting for the output is necessary or not.
        return in_type, [np.float32]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return LogPolicy(
            entropy_reg=self.entropy_reg,
            scale=self.scale)


class Risk(mx.operator.CustomOp):
    '''
    Computes the instance-level errors in terms of a given metric and propagates the errors back to other components.
    This operator needs ground truth targets to compute the risk.

    :param metric: Name of metric; currently the only supported metric is BLEU+1.
    :param ignore_ids: a list of indices to be ignored in calculating the risk
    :param eos_id: the index of the end of sequence (EOS) token in the target language
    '''
    def __init__(self, metric, ignore_ids, eos_id):
        super(Risk, self).__init__()
        self.metric = metric
        self.ignore_ids = ignore_ids
        self.eos_id = eos_id
        self.scores = None

        self.metric = metric
        if metric == 'bleu':
            self.eval_func = lambda ref, hyp: bleu.bleu1_from_counts(bleu.bleu_counts(hyp, ref))
        else:
            raise NotImplementedError()

    def forward(self, is_train, req, in_data, out_data, aux):
        action = in_data[0].asnumpy().astype(np.int32)
        target = in_data[1].asnumpy().astype(np.int32)
        is_sampled = in_data[3].asnumpy()

        # XXX compute the sentence-level (approximate) task-specific measure per instance
        batch_size = target.shape[0]
        if self.scores is None:
            # creating an internal data that holds sentence-level scores
            self.scores = mx.nd.empty((batch_size, )).as_in_context(in_data[0].context)

        if is_sampled == 0:
            # The score per instance has to be 1 if the actions are taken from the ground
            # truth indicated by the 'is_sampled' variable.
            self.scores[:] = 1
        else:
            for i in range(batch_size):
                trg = target[i]
                pred = action[i]

                # find the EOS token's position in the both sequences
                trg_eos_pos = np.where(trg == self.eos_id)[0]
                pred_eos_pos = np.where(pred == self.eos_id)[0]

                # XXX In general, ground truth sequences should contain the
                # EOS token, we check it explicitly as error handling
                if len(pred_eos_pos) > 0:
                    trg_eos_pos = trg_eos_pos[0]
                else:
                    # a virtual token (out of range) will be treated as EOS
                    # unless no EOS was sampled for the sequence.
                    trg_eos_pos = trg.shape[0]

                if len(pred_eos_pos) > 0:
                    pred_eos_pos = pred_eos_pos[0]
                else:
                    pred_eos_pos = pred.shape[0]

                self.scores[i] = self.eval_func(trg[:trg_eos_pos], pred[:pred_eos_pos])

        self.assign(out_data[0], req[0], self.scores)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        baseline = in_data[2]

        # sentence-level metrics require usually some scalar values to reduce the variance
        self.scores[:] -= baseline

        dx[:] = mx.nd.tile(mx.nd.expand_dims(self.scores, axis=dx.ndim-1), reps=(1, dx.shape[dx.ndim-1]))

        self.assign(in_grad[0], req[0], dx)


@mx.operator.register("Risk")
class RiskProp(mx.operator.CustomOpProp):
    def __init__(self, metric, ignore_ids, eos_id):
        super(RiskProp, self).__init__(need_top_grad=False)
        self.metric = metric
        self.ignore_ids = eval(ignore_ids)
        self.eos_id = eval(eos_id)

    def list_arguments(self):
        return ['data', 'label', 'baseline', 'is_sampled']

    def list_outputs(self):
        return ['score']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]        # 2d matrix
        label_shape = in_shape[1]       # 2d matrix
        baseline_shape = (data_shape[0],)
        update_state_shape = (1,)

        arguments_shp = [data_shape, label_shape,
                         baseline_shape, update_state_shape]
        score_shape = (data_shape[0],)

        return arguments_shp, [score_shape], []

    def infer_type(self, in_type):
        return in_type, [np.float32]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return Risk(self.metric, self.ignore_ids, self.eos_id)
