from argparse import ArgumentParser
from pathlib import Path
import os
import glob
import torch
import logging
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from utils import Timers

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, chunk, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs

        data_file = training_path / f"epoch_{self.data_epoch}-{chunk}.json"
        data_zip = training_path / f"epoch_{self.data_epoch}-{chunk}.zip"

        if not os.path.isfile(data_file):
            # If file not there, then there should be a zip file that extracts to it
            extract_zip(data_zip)
            assert os.path.isfile(data_file)

        logging.info('Training on: {}'.format(data_file))
        metrics_file = training_path / f"metrics_epoch_{self.data_epoch}-{chunk}.json"
        assert data_file.is_file() and metrics_file.is_file()

        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))



def get_chunks(dir_path, epoch):
    """
    Look in the specified directory for files of the form epoch_0-000, epoch_0-001, ...etc.
    and return a list of the chunks e.g. ['000', '001', '002', ...]
    There could be a mix of .json and .zip files so sometimes we could get duplicates.
    """
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)

    chunks = [x.split('-')[-1].strip('.json').strip('.zip') for x in glob.glob("{}/epoch_{}-*".format(dir_path, epoch))]
    chunks = list(set(chunks))

    return sorted(chunks)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")

    # training config
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--batch_size", default=12, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seq_length", default=256, type=int,
                        help="Seq length of each sample.")
    parser.add_argument('--train-iters', type=int, default=2000,
                       help='number of iterations per epoch')

    # distributed training config
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus. Passed from distributed launcher")


    # AMP config
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    # optimization
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # nvprof args
    parser.add_argument('--nvprof', action='store_true',
                        help='profile this program')
    parser.add_argument('--profile-start', type=int, default=200,
                        help="""Start iteration of nvidia profiler""")
    parser.add_argument('--profile-stop', type=int, default=201,
                        help="""Stop iteration of nvidia profiler""")
    parser.add_argument('--warmup-iter', type=int, default=200,
                        help="""Start iteration of nvidia profiler""")

    # benchmarking args
    parser.add_argument('--benchmark', action='store_true',
                        help='benchmark this program')
    parser.add_argument('--benchmark-dir', type=str, default="benchmark_output",
                        help="""Dir to save benchmark output stats""")
    parser.add_argument('--benchmark-start', type=int, default=1000,
                        help="""Start iteration of nvidia profiler""")
    parser.add_argument('--benchmark-stop', type=int, default=2000,
                        help="""Stop iteration of nvidia profiler""")
    parser.add_argument('--benchmark-partition', type=str, default="t4",
                        help="""Partition of gpus""")
    parser.add_argument('--log-interval', type=int, default=100,
                       help='report interval')

    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args



def main():
    args = get_args()

    total_train_examples = 0
    for i in range(args.epochs):
        chunks = get_chunks(args.pregenerated_data, i)
        if i == 0 and len(chunks) == 0:
            exit("No training data was found!")
        elif len(chunks) == 0:
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break

        for chunk in chunks:
            epoch_file = args.pregenerated_data / f"epoch_{i}-{chunk}.json"
            epoch_zip = args.pregenerated_data / f"epoch_{i}-{chunk}.zip"
            metrics_file = args.pregenerated_data / f"metrics_epoch_{i}-{chunk}.json"
            if (epoch_file.is_file() or epoch_zip.is_file()) and metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text())
                total_train_examples += metrics['num_training_examples']
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    num_train_optimization_steps = int(
        total_train_examples / args.batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // args.world_size

    # Prepare model
    model = BertForPreTraining.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler not compatible with APEX::FP16_optimizer
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)


    iteration = 0
    timers = Timers()
    benchmark_stats = defaultdict(lambda: [])
    grad_stats = defaultdict(lambda: [])

    model.train()
    for epoch in range(args.epochs):
        
        shuffled_chunks = get_chunks(args.pregenerated_data, epoch)
        random.shuffle(shuffled_chunks)
        logging.info('New shuffled chunks: {}'.format(shuffled_chunks))

        for chunk in shuffled_chunks:
            epoch_dataset = PregeneratedDataset(epoch=epoch, chunk=chunk, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                                num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.batch_size)
            data_iterator = iter(train_dataloader)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            timers('interval time').start()
            while iteration < args.train_iters:
                if args.nvprof:
                    if iteration == args.profile_start:
                        profile_cuda.profile_start()
                        print("CUDA profiling starts!")
                    if iteration == args.profile_stop:
                        profile_cuda.profile_stop()
                        print("CUDA profiling stops!")

                iteration += 1
                batch = next(data_iterator)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                outputs = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # if len(grad_stats) == 0:
                #     for name, p in model.named_parameters():
                #         if p.requires_grad:
                #             grad_stats['param_name'].append(name)
                #             grad_stats['dtype'].append(p.grad.dtype)
                #             grad_stats['size'].append(p.grad.size())
                #     df = pd.DataFrame.from_dict(grad_stats)
                #     df.to_csv("gradient_size_profile.csv")
                #     print(model.state_dict)
                #     os._exit(0)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps

                if iteration % args.log_interval == 0:
                    elapsed_time = timers('interval time').elapsed()
                    log_string = ' epoch{:2d} |'.format(epoch)
                    log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
                    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time * 1000.0 / args.log_interval)
                    log_string += ' mean loss {:.3E} |'.format(mean_loss)

                    if args.benchmark and args.rank == 0:
                        if args.benchmark_start < iteration <= args.benchmark_stop:
                            benchmark_stats['iteration'].append(iteration)
                            benchmark_stats['seq_length'].append(args.seq_length)
                            benchmark_stats['batch_size'].append(args.batch_size * args.world_size)
                            benchmark_stats['num_tokens'].append(args.seq_length * args.batch_size * args.world_size)
                            benchmark_stats['elapsed_time'].append(elapsed_time)
                            benchmark_stats['log_interval'].append(args.log_interval)
                
                    print(log_string, flush=True)

                if iteration % args.gradient_accumulation_steps == 0:
                    # scheduler.step()  # Update learning rate schedule (commented as lr_scheduler not compatible with FP16_Optimizer)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # delete break if not doing benchmarking
            break

    # Save a trained model
    if  n_gpu > 1 and args.rank == 0  or n_gpu <=1 :
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.benchmark and args.rank == 0:
        benchmark_csv = {
            k: [np.mean(l)] for k,l in benchmark_stats.items()
        }
        benchmark_csv['token_throughput'] = np.array(benchmark_csv['num_tokens']) * np.array(benchmark_csv['log_interval'])\
                                     / np.array(benchmark_csv['elapsed_time'])

        save_dir = os.path.join(
            args.benchmark_dir, 
            "{gpus}_gpus_{partition}_trials".format(
                gpus=args.world_size,
                partition=args.benchmark_partition
            )
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        df = pd.DataFrame.from_dict(benchmark_csv)
        df.to_csv(os.path.join(
            save_dir,
            "huggingface_benchmark_{partition}_batch_size_{batch_size}_seq_len_{seq_len}.csv".format(
                partition=args.benchmark_partition,
                batch_size=args.batch_size,
                seq_len=args.seq_length
            )
        ))



if __name__ == '__main__':
    main()
