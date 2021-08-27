import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
filepath = os.path.realpath(__file__)
dirpath = os.path.dirname(filepath)
for _ in range(5):
    dirpath = os.path.dirname(dirpath)
convlab_path = dirpath
import sys
sys.path.append(convlab_path)
from convlab.modules.e2e.multiwoz.Transformer.pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from convlab.modules.e2e.multiwoz.Transformer.util import get_woz_dataset
from tqdm import tqdm

import time

MODEL_PATH = "convlab/modules/e2e/multiwoz/Transformer/pytorch_transformers"

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

SPECIAL_TOKENS_V1 = [" User:", " System:", " Belief=", " Match=",  " Database=", " Ref=", " Action="]

#domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
#requestables = ['phone', 'reference', 'id', 'postcode']


MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)



def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        if name == "lm_labels":
            dataset[name] = [x + [-1] * (max_l - len(x)) for x in dataset[name]]
        elif name == "token_type_ids":
            dataset[name] = [x + [50256] * (max_l - len(x)) for x in dataset[name]]
        else:
            dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments_v1(history, reply, tokenizer, dp=[], cs=[], with_eos=True, model="gpt2", mode='train'):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    pass

def build_input_from_segments_v2(history, reply, tokenizer, dp=[], cs=[], db=[], book=[], lm_labels=True, with_eos=True, model="gpt2", mode='train'):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos = 50256, 50256

    user, system, cstok, matchtok, dbtok, booktok, dptok = [tokenizer.convert_tokens_to_ids(tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 

    instance = {}
    if mode == 'train':
        #sequence = [[bos]] + history + [[cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
        sequence = [[bos]] + history + [cstok + cs + dbtok + db + dptok + dp + booktok + book + system + reply + ([eos] if with_eos else [])]
    else:
        sequence = [[bos]] + history + [cstok + cs + db + dp + book + reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [user + s if (len(sequence)-i) % 2 else system + s for i, s in enumerate(sequence[1:-1])] + sequence[-1:]
    #print("The sequence is : ", sequence)
    l = len([i for s in sequence for i in s])

    if "gpt2" in model:
        ctx = 1024
    else:
        ctx = 512

    if l > ctx:
        print("l is ", l)
        i = 1
        while l > ctx:
            # If the sequence length is larger than 1024, we need to pop out one turn of the dialog
            d = sequence.pop(i)
            print("The poped item is ", d)
            d = sequence.pop(i)
            #logger.info("the out of lenght sequence is %s", d)
            print("The poped item is ", d)
            l -= len(d)

    instance["input_ids"] = list(chain(*sequence))
    #tokenizer.decode(instance["input_ids"])
    if mode == "train":
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db) + 2) + [dptok[0]] * (len(dp) + 2) + [booktok[0]] * (len(book) + 2) + [system[0]] * (len(reply) + 3)
    else:
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db)) + [dptok[0]] * (len(dp)) + [booktok[0]] * len(book) + [system[0]] * (len(reply))
    
    assert len(instance["token_type_ids"]) == len(instance["input_ids"])

    #logger.info(tokenizer.decode(instance["input_ids"]))
    
    if lm_labels and mode == "train":
        #print(sequence[-1])
        #print("The state is : ", tokenizer.decode(sequence[-1]))
        index_dic = {}
        for i, x in enumerate(sequence[-1][:-1]):
            index_dic[(x, sequence[-1][i+1])] = i 
        cs_index, db_index, dp_index, book_index = index_dic[tuple(cstok)], index_dic[tuple(dbtok)], index_dic[tuple(dptok)], index_dic[tuple(booktok)]
        last_sys_index = index_dic[tuple(system)]

        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] * 2 + sequence[-1][2:db_index+2] + [-1] * (dp_index - db_index - 2) + sequence[-1][dp_index:book_index+2] + [-1] * (last_sys_index - book_index - 2) + sequence[-1][last_sys_index:]
        #print("The lm labels are used and the unmasked part is : ", tokenizer.decode(sequence[-1][2:db_index+2] + sequence[-1][dp_index:book_index+2] + sequence[-1][last_sys_index:]))

        assert len(instance["lm_labels"]) == len(instance["input_ids"])

    return instance, sequence

def build_input_from_segments_v5(history, reply, tokenizer, dp=[], cs=[], db=[], lm_labels=True, with_eos=True, model="gpt2", mode='train'):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos = 50256, 50256

    user, system, cstok, matchtok, dbtok, booktok, dptok = [tokenizer.convert_tokens_to_ids(tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 

    instance = {}
    if mode == 'train':
        #sequence = [[bos]] + history + [[cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
        sequence = [[bos]] + history + [cstok + cs + dbtok + db + dptok + dp + system + reply + ([eos] if with_eos else [])]
    else:
        sequence = [[bos]] + history + [cstok + cs + db + dp + reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [user + s if (len(sequence)-i) % 2 else system + s for i, s in enumerate(sequence[1:-1])] + sequence[-1:]
    #print("The sequence is : ", sequence)
    l = len([i for s in sequence for i in s])

    if "gpt2" in model:
        ctx = 1024
    else:
        ctx = 512

    if l > ctx:
        print("l is ", l)
        i = 1
        while l > ctx:
            # If the sequence length is larger than 1024, we need to pop out one turn of the dialog
            d = sequence.pop(i)
            print("The poped item is ", d)
            d = sequence.pop(i)
            #logger.info("the out of lenght sequence is %s", d)
            print("The poped item is ", d)
            l -= len(d)

    instance["input_ids"] = list(chain(*sequence))
    
    if mode == "train":
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db) + 2) + [dptok[0]] * (len(dp) + 2) + [system[0]] * (len(reply) + 3)
    else:
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db)) + [dptok[0]] * (len(dp)) + [system[0]] * (len(reply))
    
    assert len(instance["token_type_ids"]) == len(instance["input_ids"])

    #logger.info(tokenizer.decode(instance["input_ids"]))
    
    if lm_labels and mode == "train":
        #print("The state is : ", tokenizer.decode(sequence[-1]))
        index_dic = {}
        for i, x in enumerate(sequence[-1][:-1]):
            index_dic[(x, sequence[-1][i+1])] = i 
        cs_index, db_index, dp_index= index_dic[tuple(cstok)], index_dic[tuple(dbtok)], index_dic[tuple(dptok)]

        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] * 2 + sequence[-1][2:db_index+2] + [-1] * (dp_index - db_index - 2) + sequence[-1][dp_index:] 
        #print("The lm labels are used and the unmasked part is : ", tokenizer.decode(sequence[-1][2:db_index+2] + sequence[-1][dp_index:]))

        assert len(instance["lm_labels"]) == len(instance["input_ids"])

    return instance, sequence

def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    #multiwoz_train, multiwoz_valid = get_woz_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    multiwozchat = get_woz_dataset(tokenizer, args.dataset_path, args.dataset_cache, slice_data=args.slice_data)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in multiwozchat.items():
        for name, dialog in tqdm(dataset.items()):
            for utterance in dialog["utterances"]:
                dp = utterance["dp"][0]
                cs = utterance["cs"]
                #print("CS is ", cs)
                cs = cs[0] if cs else cs
                book = utterance["book"]
                book = book[0] if book else book
                #db = utterance["db"][0] if 'v2' in args.model_version else None
                db = utterance["db"]
                db = db[0] if db else db
                #print("The db information is : ", db)
                history = utterance["history"][-(2*args.max_history+1):]
                #for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    #lm_labels = bool(j == num_candidates-1)
                candidate = utterance["candidates"][0]
                if args.model_version == 'v5':
                    instance, _ = build_input_from_segments_v5(history, candidate, tokenizer, dp, cs, db)
                else:
                    instance, _ = build_input_from_segments_v2(history, candidate, tokenizer, dp, cs, db, book, model=args.model_checkpoint)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=50256)

        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(tensor.shape)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    for x in train_dataset.tensors:
        logger.info("Every tuple in Train dataset (Batch, Seq length): {}".format(x.shape))

    logger.info("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/multiwoz/", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--model_version", type=str, default='v5', help="version of model")
    parser.add_argument("--max_history", type=int, default=30, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--adapter", type=int, default=1, help="If 1, then add adapter module to GPT2")
    parser.add_argument("--n_adapter", type=int, default=512, help="Adjust the adapter bottleneck size of GPT2")
    parser.add_argument("--log_dir", type=str, default="runs/try1_adapter", help="The dirpath for logs and checkpoints")
    parser.add_argument("--slice_data", type=bool, default=False, help="Show some cases for verification")
    args = parser.parse_args()

    args.adapter = bool(args.adapter)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        logger.info(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer

    model_class = GPT2LMHeadModel
    optimizer_class = AdamW
    #model = model_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(MODEL_PATH, using_adapter=args.adapter, using_copynet=args.adapter, n_adapter=args.n_adapter, output_attentions=True)
    #model.to(args.device)
    #optimizer = optimizer_class(model.parameters(), lr=args.lr)

    #The code for adapter is here
    if args.adapter:
        for name, param in model.named_parameters():
            if "adapter" not in name and "copylayer" not in name:
                param.requires_grad = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    model.to(args.device)

    #optimizer = optimizer_class(model.parameters(), lr=args.lr)
    if args.adapter:
        optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

    #tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, unk_token='<|unkwn|>')
    tokenizer = tokenizer_class.from_pretrained(MODEL_PATH, unk_token='<|unkwn|>')

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer

    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, *_ = model(batch[0], token_type_ids=batch[2], labels=batch[1])[0]
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)
    trainer._logger.setLevel(logging.INFO)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, lm_labels, token_type_ids = batch

            model_outputs = model(input_ids, token_type_ids=token_type_ids)
            #lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits = model_outputs[0][0]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            #return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
            return (lm_logits_flat_shifted), (lm_labels_flat_shifted)
    evaluator = Engine(inference)
    evaluator._logger.setLevel(logging.INFO)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        # Add the summarywriter for checking the status of loss decreasing
        tb_logger = TensorboardLogger(log_dir=args.log_dir)
        print("lod_dir is ", tb_logger.writer.file_writer.get_logdir())
        log_dir = tb_logger.writer.file_writer.get_logdir()
        #writer = SummaryWriter(logdir=log_dir, flush_secs=30)
        tb_logger.writer.log_dir = tb_logger.writer.file_writer.get_logdir()
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=5)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        print("Model log_dir is {}".format(tb_logger.writer.log_dir))
        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()