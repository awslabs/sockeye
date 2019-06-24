# Curriculum Learning for Neural Machine Translation

Machine translation systems based on deep neural network are expensive to train. Curriculum learning aims to address this issue by choosing the order in which samples are presented during training to help train better models faster. This documentation is a guide for how to use curriculum learning. 

This implementation makes the following assumptions:  

* The curriculum is determined by discrete complexity classes (0=easy, 1=a bit harder, 2=even harder, higher is harder).
* The curriculum will only expose the model to easy data when training starts and will gradually increase the hardness (complexity) of data.
* Shards (originally meant for data parallelism) are used to split the data by hardness (complexity).

For more technical details, please refer to our paper:

[An Empirical Exploration of Curriculum Learning for Neural Machine Translation](https://arxiv.org/pdf/1811.00739.pdf), arXiv preprint<br>
Xuan Zhang, Gaurav Kumar, Huda Khayrallah, Kenton Murray, Jeremy Gwinnup, Marianna J Martindale, Paul McNamee, Kevin Duh and Marine Carpuat<br>

There's also an application of curriculum learning on domain adaptation:

[Curriculum Learning for Domain Adaptation in Neural Machine Translation](https://www.aclweb.org/anthology/N19-1189), NAACL 2019<br>
Xuan Zhang, Pamela Shapiro, Gaurav Kumar, Paul McNamee, Marine Capuat, Kevin Duh

## Example
(1) You need to first prepare a curriculum score file `curriculum_sent.scores` ([example](https://raw.githubusercontent.com/Este1le/curriculum_learning_scores/master/scores/data_scores_ted/ted.sentence_average_rank.de.bk)). This file contains complexity scores or complexity classes in line with bitext in the training corpus. Scores should be consecutive integers, starting from 0.

(2) Prepare the training data into different shards by curriculum score file. 

```
python -m sockeye.prepare_data -s train_bpe_src \
							   -t train_bpe_trg \
							   --curriculum-score-file curriculum_sent.scores \
							   --o prepared_data	 
```

You will get sharded training data and a file `shard_scores` that assigns each shard a complexity class under the `prepared_data` directory.

(3) When training a model, turn on curriculum learning by specifying the `--curriculum-training` flag. The curriculum is updated to allow harder examples every x updates/batches. You can set the hyperparameter by `--curriculum-update-freq`.

```
python -m sockeye.train -d prepared_data \
                        --curriculum-training \
                        --curriculum-update-freq 1000 \
                        -vs valid_bpe_src \
                        -vt valid_bpe_trg \
                        -o cl_model			     
```

You will get a model trained with curriculum learning strategy.

