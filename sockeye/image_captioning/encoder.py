# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Encoders for sequence-to-sequence models.
"""
import logging
import mxnet as mx
from typing import List, Tuple

from .. import constants as C
from ..config import Config
from ..encoder import EncoderSequence, Encoder
from ..encoder import get_positional_embedding

logger = logging.getLogger(__name__)


class ImageLoadedCnnEncoderConfig(Config):
    """
    Image cnn encoder configuration. The symbolic model is loaded from disk.

    :param model_path: Path where the json file is stored.
    :param epoch: Epoch of the pre-trained model.
    :param layer_name: Name of the layer of the loaded symbol to get the encoding from.
    :param encoded_seq_len: Size of the feature layer. If the layer is a conv layer.
        encoded_seq_len should be equal to the height*width of the convolutional map,
        the number of kernel is not considered.
    :param num_embed: Number of hiddens to project the local features to.
    :param no_global_descriptor: By default the global visual feature (spatial avg of conv map)
        is concatenated to the local visual features (conv map). This option disables the use of
        the global descriptor, such that only the local ones are used.
    :param number_of_kernels: If using preextracted features, we need to know the number of dim of the features.
    :param positional_embedding_type: Which king of positional embeddingm if any.
    :param preextracted_features: Turn to bool if you preextracted featured from existing model.
    """

    def __init__(self,
                 model_path: str,
                 epoch: int,
                 layer_name: str,
                 encoded_seq_len: int,
                 num_embed: int,
                 no_global_descriptor: bool = True,
                 number_of_kernels: int = None,
                 positional_embedding_type: str = "",
                 preextracted_features: bool = False) -> None:
        super().__init__()
        self.model_path = model_path
        self.layer_name = layer_name
        self.epoch = epoch
        self.encoded_seq_len = encoded_seq_len
        self.num_embed = num_embed
        self.no_global_descriptor = no_global_descriptor
        self.number_of_kernels = number_of_kernels
        self.positional_embedding_type = positional_embedding_type
        self.preextracted_features = preextracted_features


def get_image_cnn_encoder(config: ImageLoadedCnnEncoderConfig) -> 'Encoder':
    """
    Creates a image encoder.

    :param config: Configuration for image encoder.
    :return: Encoder instance.
    """

    encoders = list()  # type: List[Encoder]
    max_seq_len = config.encoded_seq_len
    if not config.no_global_descriptor:
        max_seq_len += 1
    encoders.append(get_positional_embedding(config.positional_embedding_type,
                                             config.num_embed,
                                             max_seq_len=max_seq_len,
                                             fixed_pos_embed_scale_up_input=False,
                                             fixed_pos_embed_scale_down_positions=True,
                                             prefix=C.SOURCE_POSITIONAL_EMBEDDING_PREFIX))
    encoders.append(ImageLoadedCnnEncoder(config=config))
    return EncoderSequence(encoders)


class ImageLoadedCnnEncoder(Encoder):
    """
    Image cnn encoder. The model is loaded from disk.

    :param config: Image cnn encoder config.
    :param prefix: Name prefix for symbols of this encoder.
    """

    def __init__(self,
                 config: ImageLoadedCnnEncoderConfig,
                 prefix: str = C.CHAR_SEQ_ENCODER_PREFIX) -> None:
        self.model_path = config.model_path
        self.layer_name = config.layer_name
        self.epoch = config.epoch
        self.encoded_seq_len = config.encoded_seq_len
        self.num_embed = config.num_embed
        self.no_global_descriptor = config.no_global_descriptor
        self.preextracted_features = config.preextracted_features
        if not self.preextracted_features:
            sym, args, auxs = mx.model.load_checkpoint(self.model_path, self.epoch)
            # get layers up to layer_name
            all_layers = sym.get_internals()
            try:
                self.sym = all_layers[self.layer_name + "_output"]
            except ValueError:
                raise ValueError("Layer {} not found in the architecure located at "
                               "{}. Make sure that you choose an existing layer."\
                               .format(self.layer_name, self.model_path))
            # throws away fc weights
            self.args = dict({k: args[k] for k in args if 'fc1' not in k})
            self.auxs = auxs
            self.n_kernels = self.args[self.layer_name + "_weight"].shape[0]
            # "rename" input
            self.input = mx.sym.Variable(name=C.SOURCE_NAME)
            self.sym = self.sym(data=self.input)
        else:
            self.args = {}
            self.auxs = {}
            self.n_kernels = config.number_of_kernels
            self.sym = mx.sym.Variable(name=C.SOURCE_NAME)
        self.names = ["local_image_encoding_weight"]
        self.other_weights = {self.names[0]: mx.sym.Variable(self.names[0])}
        if not self.no_global_descriptor:
            self.names.append("global_image_encoding_weight")
            self.other_weights[self.names[1]] = mx.sym.Variable(self.names[1])
            self.encoded_seq_len += 1
        # output
        self.output_dim = self.num_embed

    def encode(self,
               data: mx.sym.Symbol,
               data_length: mx.sym.Symbol,
               seq_len: int) -> Tuple[mx.sym.Symbol, mx.sym.Symbol, int]:
        """
        Encodes data given sequence lengths of individual examples and maximum sequence length.

        :param data: Ignored. Assume that the input is the image.
        :param data_length: Vector with sequence lengths.
        :param seq_len: Maximum sequence length.
        :return: Encoded versions of input data data, data_length, seq_len.
        """

        # (batch, n_kernels, height, width) -> (batch, width, height, n_kernels)
        embedding = mx.sym.swapaxes(data=self.sym, dim1=1, dim2=3)
        # (batch, width, height, n_kernels) -> (batch, height, width, n_kernels)
        embedding = mx.sym.swapaxes(data=embedding, dim1=1, dim2=2)
        # (batch, height, width, n_kernels) -> (batch, height*width, n_kernels)
        embedding = mx.sym.Reshape(data=embedding, shape=(0, -3, self.n_kernels))
        # Feature projection layer: (batch, height*width, num_embed)
        embedding = mx.sym.FullyConnected(data=embedding, weight=self.other_weights[self.names[0]],
                                   num_hidden=self.num_embed, no_bias=True, flatten=False)
        embedding = mx.sym.Activation(data=embedding, act_type='relu')

        # Visual global description: average pooling
        if not self.no_global_descriptor:
            glob_embedding = mx.sym.mean(data=embedding, axis=1) # (batch, n_kernels)
            glob_embedding = mx.sym.FullyConnected(data=glob_embedding, weight=self.other_weights[self.names[1]],
                                       num_hidden=self.num_embed, no_bias=True)
            glob_embedding = mx.sym.Activation(data=glob_embedding, act_type='relu')
            glob_embedding = mx.sym.expand_dims(glob_embedding, axis=1)
            # Concatenate embeddings with global embedding: (batch, height*width+1, num_embed)
            embedding = mx.sym.concat(embedding, glob_embedding, dim=1, name="local_global_image_embedding")

        # Symbol to infer axis 1 dimension
        d = mx.sym.slice_axis(data=embedding, axis=2, begin=0, end=1)  # (batch, height*width, num_embed)
        d = mx.sym.clip(data=d, a_min=1.0, a_max=1.0)  # matrix of all ones
        encoded_data_length = mx.sym.sum(mx.sym.broadcast_equal(d, mx.sym.ones((1,))), axis=1)  # (batch, 1)
        encoded_data_length = mx.sym.reshape(data=encoded_data_length, shape= (-1,)) # (batch, )

        return embedding, encoded_data_length, self.encoded_seq_len

    def get_params(self):
        """
        Get the parameters of the pre-trained networks.

        :return: Tuple of arguments and auxiliaries
        """
        return self.args, self.auxs

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.output_dim

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return self.encoded_seq_len

    def get_initializers(self) -> List[Tuple[str, mx.init.Initializer]]:
        """
        Get the initializers of the network, considering the pretrained models.

        :return: List of tuples (string name, mxnet initializer)
        """
        patterns_vals = []
        # Load from args/auxs
        for k in self.args.keys():
            patterns_vals.append((k, mx.init.Load({k: self.args[k]})))
        for k in self.auxs.keys():
            patterns_vals.append((k, mx.init.Load({k: self.auxs[k]})))
        # Initialize
        for k in self.names:
            patterns_vals.append((k, mx.init.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3)))

        return patterns_vals

    def get_fixed_param_names(self) -> List[str]:
        """
        Get the fixed params of the network.

        :return: List of strings, names of the layers
        """
        args = set(self.args.keys()) | set(self.auxs.keys())

        return list(args & set(self.sym.list_arguments()))