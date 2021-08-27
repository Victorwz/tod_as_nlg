# GPT-ACN

Our code is developed on the ConvLab github page (https://github.com/ConvLab/ConvLab) and NeuralPipeline_DSTC8 (https://github.com/KAIST-AILab/NeuralPipeline_DSTC8).

## Environment setting

python version : 3.6.5

Before creating conda environment, please edit env.yml to fit on your conda root path.
For example, \'/home/jglee/anaconda\'.

```
conda env create -f env.yml
conda activate gpt_acn
```

Otherwise, you can create a conda environment with python 3.6 then install all the requirement packages.

```
conda create -n env_name python=3.6
pip install -r requirements.txt --no-dependencies
```

## How to train

The working directory is $ROOT/Convlab.
The description below follows the working directory.

```
cd ConvLab # (working directory)
cd data/multiwoz
unzip total.zip
unzip val.zip
cd ../../  # (working directory)
sh run.sh
```

`-m torch.distributed.launch --nproc_per_node=${#OfGPUs}` in run.sh is to use multi GPUs. If you would like to train with single GPU, you can remove this part.

The parameter of `--log_dir` is the path for saving trained model checkpoints. You need to modify this parameter before training.

Please refer to huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai.) 


## How to test on ConvLab

In convlab/modules/e2e/multiwoz/Transformer/Transformer.py, the Transformer class manages our algorithm.

```
python run.py submission.json submission4 eval
```

We have pre-set the path to model checkpoint in 'model_checkpoint'. If you want to evaluate your own fine-tuned weights, please handle the "model_checkpoint" on the right submission name (e.g. submission4) in 'convlab/spec/submission.json'.

## Credit

Our code is based on huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai.) and NeuralPipeline_DSTC8 (https://github.com/KAIST-AILab/NeuralPipeline_DSTC8).

