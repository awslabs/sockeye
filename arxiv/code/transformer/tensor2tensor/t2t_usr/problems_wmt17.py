import tensor2tensor as t2t
import os
from tensor2tensor.data_generators.problem import Problem, SpaceID
from tensor2tensor.data_generators.translate import TranslateProblem, token_generator
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base
from tensor2tensor.data_generators import text_encoder

EOS = text_encoder.EOS_ID

@registry.register_hparams
def transformer_wmt17_base():
  # transformer v2
  hparams = transformer_base()
  return hparams

"""
EN-DE
"""
ENDE_TRAIN_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.bpe.%s' % os.environ['SOURCE'])
ENDE_TRAIN_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.bpe.%s' % os.environ['TARGET'])
ENDE_DEV_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.bpe.%s' % os.environ['SOURCE'])
ENDE_DEV_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.bpe.%s' % os.environ['TARGET'])

@registry.register_problem
class Wmt17EnDeBpe32k(TranslateProblem):
  """t2t-bpe model that builds t2t subwords on top of already BPE-ed data"""

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe.ende" 

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_txt_vocab(
        data_dir, self.vocab_file, self.targeted_vocab_size, filepatterns=[ENDE_TRAIN_SRC, ENDE_TRAIN_TRG])
    if train:
      data_src = ENDE_TRAIN_SRC
      data_trg = ENDE_TRAIN_TRG
    else:
      data_src = ENDE_DEV_SRC
      data_trg = ENDE_DEV_TRG
    return token_generator(data_src, data_trg, symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return SpaceID.DE_TOK

ENDE_TRAIN_TOK_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.clean.%s' % os.environ['SOURCE'])
ENDE_TRAIN_TOK_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.clean.%s' % os.environ['TARGET'])
ENDE_DEV_TOK_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.tok.%s' % os.environ['SOURCE'])
ENDE_DEV_TOK_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.tok.%s' % os.environ['TARGET'])

@registry.register_problem
class Wmt17EnDe32k(TranslateProblem):
  """t2t-tok model using only tokenized data for comparison"""

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.ende" 

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_txt_vocab(
        data_dir, self.vocab_file, self.targeted_vocab_size, filepatterns=[ENDE_TRAIN_TOK_SRC, ENDE_TRAIN_TOK_TRG])
    if train:
      data_src = ENDE_TRAIN_TOK_SR
      data_trg = ENDE_TRAIN_TOK_TRG
    else:
      data_src = ENDE_DEV_TOK_SRC
      data_trg = ENDE_DEV_TOK_TRG
    return token_generator(data_src, data_trg, symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return SpaceID.DE_TOK


"""
LV-EN
"""
LVEN_TRAIN_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.bpe.%s' % os.environ['SOURCE'])
LVEN_TRAIN_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'train.tok.bpe.%s' % os.environ['TARGET'])
LVEN_DEV_SRC = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.bpe.%s' % os.environ['SOURCE'])
LVEN_DEV_TRG = os.path.join(os.environ['DATADIR'], os.environ['PAIR'], 'dev.bpe.%s' % os.environ['TARGET'])

@registry.register_problem
class Wmt17LvEnBpe32k(TranslateProblem):
  """t2t-bpe model"""

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe.ende"

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_txt_vocab(
        data_dir, self.vocab_file, self.targeted_vocab_size, filepatterns=[LVEN_TRAIN_SRC, LVEN_TRAIN_TRG])
    if train:
      data_src = LVEN_TRAIN_SRC
      data_trg = LVEN_TRAIN_TRG
    else:
      data_src = LVEN_DEV_SRC
      data_trg = LVEN_DEV_TRG
    return token_generator(data_src, data_trg, symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return SpaceID.DE_TOK
