# Learning Light-Weight Translation Models from Deep Transformer

> Bei Li, Ziyang Wang, Hui Liu, Quan Du, Tong Xiao, Chunliang Zhang, Jingbo Zhu. In Proceedings of AAAI, 2021. [\[paper\]](https://arxiv.org/pdf/2012.13866.pdf)

# GPKD Method on Fairseq
The  GPKD method is based on the Transformer system Fairseq v0.6.2 implemented by Facebook

## Runtime Environment
This system has been tested in the following environment.
+ Python version \>=3.6
+ Pytorch version \>=1.0.0

### Group-Permutation Training:

First, we train the teacher network with group-permutation training strategy which rectifies the information flow.

For the `--arch` and  arguments, `group_` should be used as the prefix for teacher network, such as:
`--arch transformer_t2t_wmt_en_de` -> `--arch group_transformer_t2t_wmt_en_de`

We can set the detailed arguments of different architectures in `group_transformer.py`

Example of the script for training phase:
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
max_token=2048 
data_dir=google 
save_dir_1=
python3 -u train.py data-bin/$data_dir \
--distributed-world-size 8 -s en -t de \
--ddp-backend no_c10d \
--arch group_transformer_t2t_wmt_en_de \
--optimizer adam --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 16000 \
--lr $lr_1 --min-lr 1e-09 \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens $max_tokens \
--update-freq 4 \
--no-progress-bar \
--fp16 \
--adam-betas '(0.9, 0.997)' \
--log-interval 100 \
--share-all-embeddings \
--max-epoch 21 \
--save-dir $save_dir_1 \
--keep-last-epochs 5 \
--tensorboard-logdir $save_dir_1 > $save_dir_1/train.log
```

We can use the  group-permutation method to train a teacher network  by `train.sh`

```
sh train.sh
```
### Generating SKD Data

Given the training dataset \{ X, Y \},  the teacher network translates the source inputs into the target Z. Then the SKD data is the collection of  \{ X, Z \}.

We can set the save dir of the teacher model and the other arguments to generate the target of SKD Data  by `translation.sh`.

```
sh translation.sh
```

### Student Training

First, we need extract the student network form the teacher network by `extract.py`, we can set the index of the encoder layers or decoder layers that will be extracted.

```
python3 extract.py teacher_dir/checkpoint_last.pt student_dir/checkpoint_last.pt
```
Then, we can finetune the student network with the SKD data  by `train.sh`

```
sh train.sh
```

But we need reset the `--arch`, `--save-dir` and `--max_epoch`

The architecture of student model is Transformer, so we reset `--arch`:
`--arch group_transformer_t2t_wmt_en_de` -> `--arch transformer_t2t_wmt_en_de`

## Skipping Sub-Layers Method

To further enhance the teacher model, we present a Skipping Sub-Layer method to randomly omit sub-layers to introduce perturbation into training.

For the `--arch` and  arguments, `skipping_sublayer_` should be used as the prefix for teacher network, such as:
`--arch transformer_t2t_wmt_en_de` -> `--arch skipping_sublayer_transformer_t2t_wmt_en_de`

We can set the detailed arguments of different architecture in `skipping_sublayer_transformer.py`.

We can use the  Skipping Sub-Layer method method to train a network  by `train.sh`
```
sh train.sh
```

## Citation
please cite as:
```bibtex
@article{li2020shallow,
  title={Shallow-to-Deep Training for Neural Machine Translation},
  author={Li, Bei and Wang, Ziyang and Liu, Hui and Jiang, Yufan and Du, Quan and Xiao, Tong and Wang, Huizhen and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2010.03737},
  year={2020}
}
```

<p align="center">
  <img src="./OR-Transformer/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

### What's New:

- March 2020: [Byte-level BPE code released](examples/byte_level_bpe/README.md)
- February 2020: [mBART model and code released](examples/mbart/README.md)
- February 2020: [Added tutorial for back-translation](https://github.com/pytorch/fairseq/tree/master/examples/backtranslation#training-your-own-model-wmt18-english-german)
- December 2019: [fairseq 0.9.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.9.0)
- November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
- November 2019: [CamemBERT model and code released](examples/camembert/README.md)
- November 2019: [BART model and code released](examples/bart/README.md)
- November 2019: [XLM-R models and code released](examples/xlmr/README.md)
- September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
- August 2019: [WMT'19 models released](examples/wmt19/README.md)
- July 2019: fairseq relicensed under MIT license
- July 2019: [RoBERTa models and code released](examples/roberta/README.md)
- June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

### Features:

Fairseq provides reference implementations of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  - [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  - [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  - [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  - [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- **LightConv and DynamicConv models**
  - [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
- **Long Short-Term Memory (LSTM) networks**
  - Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
- **Transformer (self-attention) networks**
  - Attention Is All You Need (Vaswani et al., 2017)
  - [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  - [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  - [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/transformer_lm/README.md)
  - [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  - [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  - [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  - [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  - [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  - [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
- **Non-autoregressive Transformers**
  - Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  - Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  - Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  - Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  - [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)


**Additionally:**
- multi-GPU (distributed) training on one machine or across multiple machines
- fast generation on both CPU and GPU with multiple search algorithms implemented:
  - beam search
  - Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  - sampling (unconstrained, top-k and top-p/nucleus)
- large mini-batch training even on a single GPU via delayed updates
- mixed precision training (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
- extensible: easily register new models, criterions, tasks, optimizers and learning rate schedulers

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:
```python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```
See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

![Model](fairseq.gif)

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

To install fairseq:
```bash
pip install fairseq
```

On MacOS:
```bash
CFLAGS="-stdlib=libc++" pip install fairseq
```

If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

**Installing from source**

To install fairseq from source and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
```

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

- [Translation](examples/translation/README.md): convolutional and transformer models are available
- [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available
- [wav2vec](examples/wav2vec/README.md): wav2vec large model is available

We also have more detailed READMEs to reproduce results from specific papers:
- [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
- [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
- [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
- [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
- [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
- [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
- [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
- [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
- [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
- [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
- [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
- [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License
fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@article{li2020learning,
  title={Learning Light-Weight Translation Models from Deep Transformer},
  author={Li, Bei and Wang, Ziyang and Liu, Hui and Du, Quan and Xiao, Tong and Zhang, Chunliang and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2012.13866},
  year={2020}
}
```