# Task-Oriented Dialogue System as Natural Language Generation

Official implementation of SIGIR 2022 Paper "[Task-Oriented Dialogue System as Natural Language Generation](https://arxiv.org/abs/2108.13679)". Please cite our paper if you find this repository helpful in your research:
```
@article{wang2021task,
  title={Task-Oriented Dialogue System as Natural Language Generation},
  author={Wang, Weizhi and Zhang, Zhirui and Guo, Junliang and Dai, Yinpei and Chen, Boxing and Luo, Weihua},
  journal={arXiv preprint arXiv:2108.13679},
  year={2021}
}
```

## Environment setting

python version : 3.6.5

Before creating conda environment, please edit env.yml to fit on your conda root path.
For example, \'/home/weizhi.wwz/anaconda\'.

```
conda env create -f env.yml
conda activate gpt_acn
```

Otherwise, you can create a conda environment with python 3.6 then install all the requirement packages.

```
conda create -n env_name python=3.6 # revise env_name as you like
conda activate env_name
pip install -r requirements.txt --no-dependencies
```

## How to train

The working directory is $ROOT/Convlab.
The description below follows the working directory.

```
cd ConvLab # (working directory)
bash run.sh
```

`-m torch.distributed.launch --nproc_per_node=${#GPU}` in run.sh is to use multi GPUs. If you would like to train with single GPU, you can remove this part.

The parameter of `--log_dir` is the path for saving trained model checkpoints. You need to modify this parameter before training.

Please refer to huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai.)


## How to test on ConvLab

In convlab/modules/e2e/multiwoz/Transformer/Transformer.py, the Transformer class manages our algorithm.

```
python run.py submission.json submission4 eval
```

We have pre-set the path to model checkpoint in 'model_checkpoint'. If you want to evaluate your own fine-tuned weights, please handle the "model_checkpoint" on the right submission name (e.g. submission4) in 'convlab/spec/submission.json'.

## Credit

Our code is developed on the ConvLab github page (https://github.com/ConvLab/ConvLab), huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai) and NeuralPipeline_DSTC8 (https://github.com/KAIST-AILab/NeuralPipeline_DSTC8).

