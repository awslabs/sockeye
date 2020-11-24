import random
from typing import Optional

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io


def get_segments(mask_len, span_len):
    """Generate the masked segments up to the mask length. """
    segs = []
    while mask_len >= span_len:
        segs.append(span_len)
        mask_len -= span_len
    if mask_len != 0:
        segs.append(mask_len)
    return segs


def get_segments_poisson(mask_len, span_len):
    """ Generate the masked segments up to the mask length with span lengths sampled according to a poisson distribution, similar to BART. """
    segs = []
    while mask_len > 0:
        curr_span_len = np.random.poisson(lam=span_len)
        curr_span_len = min(mask_len, max(1, curr_span_len))
        segs.append(curr_span_len)
        mask_len -= curr_span_len
    if mask_len != 0:
        segs.append(mask_len)
    return segs


def shuffle_segments(segs, unmasked_tokens):
    """
    We control 20% mask segment is at the start of sentences
                20% mask segment is at the end   of sentences
                60% mask segment is at random positions,
    """
    # TODO: base this on an instance of a random number generator to base this on the seed

    p = np.random.random()
    if p >= 0.8:
        shuf_segs = segs[1:] + unmasked_tokens
    elif p >= 0.6:
        shuf_segs = segs[:-1] + unmasked_tokens
    else:
        shuf_segs = segs + unmasked_tokens

    random.shuffle(shuf_segs)

    if p >= 0.8:
        shuf_segs = segs[0:1] + shuf_segs
    elif p >= 0.6:
        shuf_segs = shuf_segs + segs[-1:]
    return shuf_segs


def unfold_segments(segs, start=1):
    """Unfold the random mask segments, for example:
        The shuffle segment is [2, 0, 0, 2, 0],
        so the masked segment is like:
        [1, 1, 0, 0, 1, 1, 0]
        [1, 2, 3, 4, 5, 6, 7] (positions)
        (1 means this token will be masked, otherwise not)
        We return the position of the masked tokens like:
        [1, 2, 5, 6]
    """
    pos = []
    curr = start  # We (optionally) do not mask the start token
    for l in segs:
        if l >= 1:
            pos.extend([curr + i for i in range(l)])
            curr += l
        else:
            curr += 1
    return pos


def mask_word(data: mx.nd.NDArray, pred_probs: mx.nd.NDArray, num_vocab: Optional[int], mask_index: int):
    """[summary]

    :param data: (seq_len,)
    :param pred_probs: 3d: (mask_prob, real_prob, rand_prob)
    :param num_vocab: [description]
    :param mask_index: [description]
    :return: [description]
    :rtype: [type]
    """

    if num_vocab is None:
        data_real = data
        data_mask = mx.nd.full(data.shape, val=mask_index, dtype=data.dtype, ctx=data.context)

        choice = mx.nd.random.multinomial(pred_probs, shape=len(data))

        # TODO: alternatively concat and then take!?
        # TODO: try masked addition?!
        # _w = _w_mask * (choice == 0).numpy() + _w_real * (choice == 1).numpy() + _w_rand * (choice == 2).numpy()
        # TODO: this could be simplified to just a single condition
        out_data = mx.nd.where(
            condition=(choice == 0),
            x=data_mask,
            y=mx.nd.where(
                condition=(choice == 1),
                x=data_real,
                y=data_mask
            )
        )
        return out_data
    else:

        # data_real = data.copy()
        data_real = data
        # We sample words at random (excluding special symbols)
        data_rand = mx.nd.random.randint(len(C.VOCAB_SYMBOLS), num_vocab, shape=data.shape, dtype=data.dtype,
                                         ctx=data.context)
        data_mask = mx.nd.full(data.shape, val=mask_index, dtype=data.dtype, ctx=data.context)

        choice = mx.nd.random.multinomial(pred_probs, shape=len(data))

        # TODO: alternatively concat and then take!?
        # TODO: try masked addition similar to MASS code?!
        # _w = _w_mask * (choice == 0).numpy() + _w_real * (choice == 1).numpy() + _w_rand * (choice == 2).numpy()
        out_data = mx.nd.where(
            condition=(choice == 0),
            x=data_mask,
            y=mx.nd.where(
                condition=(choice == 1),
                x=data_real,
                y=data_rand
            )
        )
        return out_data


def mask_bart_flat(data: mx.nd.NDArray, data_length: mx.nd.NDArray, pred_probs: mx.nd.NDArray,
                   vocab_size: Optional[int] = None, mask_index: int = C.MASK_ID,
                   span_len: int = 8, span_len_distribution=None,
                   word_mass: float = 0.5, vocab=None, mask_eos=True, word_mass_sampled=False):
    """ Mask the data.

    :param data: (batch_size, seq_len, 1)
    :param span_len: The size of the spans that are masked.
    :param word_mass: The percent to words to mask per sentence.
    :return: source, target, target_pos.
    """
    source = data.copy()

    positions = []
    for i in range(data.shape[0]):
        sent_length = data_length[i].asscalar()

        if mask_eos:
            mask_sent_length = sent_length
        else:
            mask_sent_length = sent_length - 1

        if mask_sent_length == 1:
            # single tokens we always 'mask'
            mask_len = 1
        else:
            if word_mass_sampled:
                curr_word_mass = random.random() * word_mass
            else:
                curr_word_mass = word_mass
            mask_len = int(round(mask_sent_length * curr_word_mass))
        # Note: MASS originally used start=1 (to not mask their BOS symbol, but we don't have a BOS symbol)
        start = 0
        unmasked_tokens = [0 for _ in range(mask_sent_length - mask_len - start)]
        # TODO: optionally sample the span_len instead of taking a fixed length
        if span_len_distribution is None:
            segs = get_segments(mask_len, span_len)
        elif span_len_distribution == "poisson":
            segs = get_segments_poisson(mask_len, span_len)
        else:
            raise ValueError("unknown")
        shuf_segs = shuffle_segments(segs, unmasked_tokens)
        pos_i = unfold_segments(shuf_segs, start=start)
        # Take the global index into the flat version of source:
        positions.extend([i * data.shape[1] + p for p in pos_i])

    if len(positions) == 0:
        # If nothing is to be masked we can just return source as is
        return source
    positions = mx.nd.array(positions, dtype=np.int32, ctx=source.context)
    source = source.reshape(shape=(-1,))
    source[positions] = mask_word(source[positions], pred_probs, vocab_size, mask_index)
    source = source.reshape(shape=data.shape)

    return source


def mask_dae_flat(data: mx.nd.NDArray, data_length: mx.nd.NDArray, pdrop=0.1, k=3):
    """ We apply the noising by Lample et al. (2017), namely: dropping and shuffling words.
    """
    source = mx.nd.zeros(shape=(np.prod(data.shape),), ctx=data.context, dtype=np.int32)

    positions = []
    words = []
    for i in range(data.shape[0]):
        sent_length = data_length[i].asscalar()

        sent = data[i, :sent_length].reshape(-1).asnumpy().tolist()
        # randomly drop words, don't drop EOS:
        sent = [w for w in sent if w == sent[-1] or random.random() > pdrop]
        # Sligthly change the sentence order:
        sent = [w for _, w in sorted((p + random.random() * k + 1, w) for p, w in enumerate(sent[:-1]))] + [sent[-1]]
        words.extend(sent)
        positions.extend([i * data.shape[1] + p for p in range(0, len(sent))])

    positions = mx.nd.array(positions, dtype=np.int32, ctx=source.context)
    words = mx.nd.array(words, dtype=np.int32, ctx=source.context)
    source[positions] = words
    source = source.reshape(shape=data.shape)

    return source


def mask_mass(data: mx.nd.NDArray, data_length: mx.nd.NDArray, pred_probs: mx.nd.NDArray, vocab_size: int,
              mask_index: int, span_len: int = 8, word_mass: float = 0.5, vocab=None, mask_eos=True):
    """ Mask the data.

    :param data: (batch_size, seq_len, 1)
    :param span_len: The size of the spans that are masked.
    :param word_mass: The percent to words to mask per sentence.
    :return: source, target, target_pos.
    """
    source = data.copy()
    squeezed_data = mx.nd.reshape(data, shape=(0, 0))

    # Target is the data but replacing the EOS with a BOS at the beginning of the sentence
    target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
    target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
    # target: (batch_size, seq_len, 1)
    target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

    all_positions, all_targets, all_labels = [], [], []

    flat_eos_indices = []

    max_len = -1

    for i in range(data.shape[0]):
        sent_length = data_length[i].asscalar()

        mask_sent_length = sent_length
        if mask_sent_length < 2:
            # We expect at least one token and EOS
            assert False
        else:
            mask_len = int(round(mask_sent_length * word_mass))
            start = 0
            unmasked_tokens = [0 for _ in range(mask_sent_length - mask_len - start)]
            segs = get_segments(mask_len, span_len)
            shuf_segs = shuffle_segments(segs, unmasked_tokens)
            pos_i = unfold_segments(shuf_segs, start=start)
            all_positions.append(pos_i)

            flat_eos_indices.append(i * data.shape[1] + sent_length - 1)

            if len(pos_i) > max_len:
                max_len = len(pos_i)

    flat_indices_from = []
    flat_indices_to = []
    positions = []
    for i, pos_i in enumerate(all_positions):
        flat_indices_from.extend([i * data.shape[1] + p for p in pos_i])
        flat_indices_to.extend([i * max_len + j for j, _ in enumerate(pos_i)])
        positions.extend(pos_i)

    flat_indices_from = mx.nd.array(flat_indices_from, dtype=np.int32)
    flat_indices_to = mx.nd.array(flat_indices_to, dtype=np.int32)
    positions = mx.nd.array(positions, dtype=np.int32)
    flat_eos_indices = mx.nd.array(flat_eos_indices, dtype=np.int32)

    source = source.reshape(shape=(-1,))
    source[flat_indices_from] = mask_word(source[flat_indices_from], pred_probs, vocab_size, mask_index)
    # Optionally restore EOS symbols instead of masks if EOS bad been masked (the reason to not exclude it from masking
    # in the first place is that we want EOS to still appear on the target side)
    if not mask_eos:
        source[flat_eos_indices] = C.EOS_ID
    source = source.reshape(shape=data.shape)

    target_new = mx.nd.full(shape=(data.shape[0] * max_len), val=C.PAD_ID, dtype=np.int32)
    labels_new = mx.nd.full(shape=(data.shape[0] * max_len), val=C.PAD_ID, dtype=np.int32)
    positions_new = mx.nd.full(shape=(data.shape[0] * max_len), val=C.PAD_ID, dtype=np.int32)

    target_flat = target.reshape((-1,))
    target_new[flat_indices_to] = target_flat[flat_indices_from]
    target_new = target_new.reshape((data.shape[0], max_len))

    squeezed_data_flat = squeezed_data.reshape((-1,))
    labels_new[flat_indices_to] = squeezed_data_flat[flat_indices_from]
    labels_new = labels_new.reshape((data.shape[0], max_len))

    positions_new[flat_indices_to] = positions
    positions_new = positions_new.reshape(shape=(data.shape[0], max_len))

    return source, target_new, labels_new, positions_new


def mask_data(data, data_length, vocab_size: Optional[int] = None, vocab=None, style: str = "BART", word_mass=0.5,
              word_mass_sampled=False):
    if style == "BART":
        pred_probs = mx.nd.array([0.8, 0.05, 0.15])
        data = mask_bart_flat(data, data_length, pred_probs=pred_probs, vocab_size=vocab_size, mask_index=C.MASK_ID,
                              vocab=vocab, word_mass=word_mass, word_mass_sampled=word_mass_sampled)
    elif style == "BART_NOMASKEOS":
        pred_probs = mx.nd.array([0.8, 0.05, 0.15])
        data = mask_bart_flat(data, data_length, pred_probs=pred_probs, vocab_size=vocab_size, mask_index=C.MASK_ID,
                              vocab=vocab, mask_eos=False, word_mass=word_mass, word_mass_sampled=word_mass_sampled)
    elif style == "BART_NOMASKEOS_NORANDWORD":
        pred_probs = mx.nd.array([0.9, 0.1, 0.0])
        data = mask_bart_flat(data, data_length, pred_probs=pred_probs, vocab_size=vocab_size, mask_index=C.MASK_ID,
                              vocab=vocab, mask_eos=False, word_mass=word_mass, word_mass_sampled=word_mass_sampled)
    elif style == "BART_NOMASKEOS_POISSON":
        pred_probs = mx.nd.array([0.8, 0.05, 0.15])
        data = mask_bart_flat(data, data_length, pred_probs=pred_probs,
                              vocab_size=vocab_size, mask_index=C.MASK_ID,
                              span_len=4, span_len_distribution="poisson",
                              vocab=vocab, mask_eos=False, word_mass=word_mass, word_mass_sampled=word_mass_sampled)
    elif style == "BART_NOMASKEOS_POSSION_NORANDWORD":
        pred_probs = mx.nd.array([0.9, 0.1, 0.])
        data = mask_bart_flat(data, data_length, pred_probs=pred_probs,
                              vocab_size=vocab_size, mask_index=C.MASK_ID,
                              span_len=3, span_len_distribution="poisson",
                              vocab=vocab, mask_eos=False, word_mass=word_mass, word_mass_sampled=word_mass_sampled)
    elif style == "DAE":
        data = mask_dae_flat(data, data_length)
    elif style == "BART_PAD":
        raise NotImplemented()
    elif style == "BART_PAD_MASK":
        raise NotImplemented()
    elif style == "MASS":
        raise NotImplemented()
    elif style == "MASS_NOMASKEOS":
        raise NotImplemented()
    else:
        raise ValueError(f"Unknown style {style}")
    return data


def mono_batch_to_parallel_batch(batch: 'data_io.MonoBatch', vocab_size: int, vocab, style: str = "BART",
                                 word_mass=0.5,
                                 word_mass_sampled=False) -> 'data_io.Batch':
    """ Converts a monolingual batch to a parallel batch through by masking the source and predicting the full target. """
    # note: the data has EOS but no BOS
    # TODO: directly expose data as int in the monolingual data iterator!

    source_lang = batch.lang
    target_lang = batch.lang

    # mx.nd.full((1,), val=self.lang_vocab[self.source_lang])
    if style == "BART":
        # BART: The BART style 'noises' the input, but uses the full sequence as the target (including tokens not masked on the source side)

        # note: the data has EOS but not BOS
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        squeezed_data = mx.nd.reshape(data, shape=(0, 0))
        label = squeezed_data
        target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
        target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
        # target: (batch_size, seq_len, 1)
        target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

        # Creating a fake target factor:
        # TODO: full target factor support
        target = mx.nd.reshape(target, shape=(0, 0, 1))
        label = mx.nd.reshape(target, shape=(0, 0, 1))

        source = mask_data(data, data_length, vocab_size=vocab_size, vocab=vocab, style=style, word_mass=word_mass,
                           word_mass_sampled=word_mass_sampled)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          label.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang)

        return batch
    if style == "DAE":
        # BART: The BART style 'noises' the input, but uses the full sequence as the target (including tokens not masked on the source side)

        # note: the data has EOS but not BOS
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        squeezed_data = mx.nd.reshape(data, shape=(0, 0))
        label = squeezed_data
        target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
        target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
        # target: (batch_size, seq_len, 1)
        target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

        source = mask_data(data, data_length, vocab_size=vocab_size, vocab=vocab, style=style, word_mass=word_mass,
                           word_mass_sampled=word_mass_sampled)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          label.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang)

        return batch
    if style == "BART_NOMASKEOS":
        # BART: The BART style 'noises' the input, but uses the full sequence as the target (including tokens not masked on the source side)

        # note: the data has EOS but not BOS
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        squeezed_data = mx.nd.reshape(data, shape=(0, 0))
        label = squeezed_data
        target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
        target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
        # target: (batch_size, seq_len, 1)
        target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

        source = mask_data(data, data_length, vocab_size=vocab_size, vocab=vocab, style=style, word_mass=word_mass,
                           word_mass_sampled=word_mass_sampled)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          label.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang)

        return batch
    if style == "BART_NOMASKEOS_NORANDWORD":
        # BART: The BART style 'noises' the input, but uses the full sequence as the target (including tokens not masked on the source side)

        # note: the data has EOS but not BOS
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        squeezed_data = mx.nd.reshape(data, shape=(0, 0))
        label = squeezed_data
        target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
        target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
        # target: (batch_size, seq_len, 1)
        target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

        source = mask_data(data, data_length, vocab_size=vocab_size, vocab=vocab, style=style, word_mass=word_mass,
                           word_mass_sampled=word_mass_sampled)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          label.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang)

        return batch
    if style == "BART_NOMASKEOS_POISSON" or "BART_NOMASKEOS_POSSION_NORANDWORD":
        # BART: The BART style 'noises' the input, but uses the full sequence as the target (including tokens not masked on the source side)

        # note: the data has EOS but not BOS
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        squeezed_data = mx.nd.reshape(data, shape=(0, 0))
        label = squeezed_data
        target = squeezed_data[:, :-1]  # skip last column (for longest-possible sequence, this already removes <eos>)
        target = mx.nd.where(target == C.EOS_ID, mx.nd.zeros_like(target), target)  # replace other <eos>'s with <pad>
        # target: (batch_size, seq_len, 1)
        target = mx.nd.concat(mx.nd.full((target.shape[0], 1), val=C.BOS_ID, dtype=np.int32), target)

        source = mask_data(data, data_length, vocab_size=vocab_size, vocab=vocab, style=style, word_mass=word_mass,
                           word_mass_sampled=word_mass_sampled)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          label.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang)

        return batch
    elif style == "MASS":
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        pred_probs = mx.nd.array([0.8, 0.05, 0.15])

        assert word_mass_sampled is False, "Not supported."

        source, target, labels, positions = mask_mass(data, data_length, pred_probs=pred_probs, vocab_size=vocab_size,
                                                      mask_index=C.MASK_ID, vocab=vocab, mask_eos=True,
                                                      word_mass=word_mass)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          labels.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang, target_steps=positions)

        return batch
    elif style == "MASS_NOMASKEOS":
        data = batch.data.astype(np.int32)
        data_length = batch.data_length.astype(np.int32)
        pred_probs = mx.nd.array([0.8, 0.05, 0.15])

        assert word_mass_sampled is False, "Not supported."

        source, target, labels, positions = mask_mass(data, data_length, pred_probs=pred_probs, vocab_size=vocab_size,
                                                      mask_index=C.MASK_ID, vocab=vocab, mask_eos=False,
                                                      word_mass=word_mass)

        batch = data_io.create_batch_from_parallel_sample(source.astype(np.float32), target.astype(np.float32),
                                                          labels.astype(np.float32), source_lang=source_lang,
                                                          target_lang=target_lang, target_steps=positions)

        return batch
    else:
        raise ValueError(f"Unknown style {style}")
