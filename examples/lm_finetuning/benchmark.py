import os
import subprocess as sp
import pandas as pd
import shlex
import argparse
import json
import glob
import time


class Benchmarking():
    def __init__(self, cmds, nodes=1, max_load=4):
        assert len(cmds) % nodes == 0
        assert nodes >= 1
        self.sleep_time = 60
        self.max_load = max_load
        self.nodes = nodes
        self.scheduled = []
        self._add_cmds(cmds)

    def _add_cmds(self, cmds):
        self.cmds = []
        for x in cmds:
            if isinstance(x, str):
                self.cmds.append(shlex.split(x))
            else:
                self.cmds.append(x)

    def _wait(self):
        scheduled = self.scheduled[:]
        while True:
            for proc in scheduled:
                if proc.poll() is not None:
                    self.scheduled.remove(proc)
                    return

            print("Waiting for somejob to finish!")
            time.sleep(self.sleep_time)

    def schedule(self):
        while self.cmds:
            # submit distributed jobs together
            for _ in range(self.nodes):
                cmd = self.cmds.pop()
                if self.max_load > 0:
                    while len(self.scheduled) >= self.max_load:
                        self._wait()

                print("Scheduling new job to Slurm")
                self.create_subprocess(cmd)
                # slurm would prevent submitting too many jobs simutaniously
                time.sleep(3)
            # one patch of distributed job submitted, sleep
            time.sleep(self.sleep_time)

        for proc in self.scheduled:
            proc.wait()
    
        print("Slurm scheduling done!")

    def create_subprocess(self, cmd):
        # if queued => srun: job xxxx queued and waiting for resources
        try:
            proc = sp.Popen(cmd)
            # output = sp.check_output(cmd)
        except sp.CalledProcessError as e:
            print("Failed to create process on Slurm! (exit code: {})".format(e.returncode))
            return

        self.scheduled.append(proc)


def combined_csv(args):
    nodes = args.world_size/args.num_gpu_per_node
    world_size = args.world_size
    partition=args.partition

    glob_pattern = os.path.join(
        args.benchmark_dir, 
        "{nodes}_nodes_{gpus}_gpus_{partition}_trials".format(nodes=nodes, gpus=world_size, partition=partition), 
        "*.csv")
    csv_files = [f for f in glob.glob(glob_pattern)]

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])

    # export to csv
    export_csv = os.path.join(
        args.benchmark_dir, 
        "{partition}_num_gpu_{gpus}_benchmark.csv".format(
            partition=partition,
            gpus=world_size
        )
    )
    combined_csv.to_csv(export_csv, index=False, encoding='utf-8-sig')


def has_ran(args, batch_size, seq_len):
    save_dir = os.path.join(
        args.benchmark_dir, 
        "{gpus}_gpus_{partition}_trials".format(
            gpus=args.world_size,
            partition=args.partition
        )
    )
    file_path = os.path.join(save_dir,
        "huggingface_benchmark_{partition}_batch_size_{batch_size}_seq_len_{seq_len}.csv".format(
            partition=args.partition,
            batch_size=batch_size,
            seq_len=seq_len))
    return os.path.isfile(file_path)


# single gpu training
def baseline(args):
    data = "/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/"
    cmd = """srun --mem=12G -c 4 --gres=gpu:1 -t 0 -p {partition} 
             python finetune_on_pregenerated.py \
                --pregenerated_data {data} \
                --bert_model bert-large-uncased \
                --do_lower_case \
                --output_dir finetuned_lm/ \
                --epochs 1 \
                --fp16 \
                --batch_size {batch_size} \
                --seq_length {seq_len} \
                --benchmark \
                --benchmark_dir {benchmark_dir} \
                --benchmark_partition {benchmark_partition}"""

    # seq_len : batch size
    batch_config = {
        128: [1, 4, 8, 12, 16, 20, 22],
    }

    cmds = []
    for seq_len, batch_sizes in batch_config.items():
        for x in batch_sizes:
            if not has_ran(args, x, seq_len):
                cmds.append(cmd.format(
                    partition=args.partition,
                    benchmark_partition=args.partition,
                    batch_size=x,
                    seq_len=seq_len,
                    data=data,
                    benchmark_dir=args.benchmark_dir
                ))

    benchmark = Benchmarking(cmds, max_load=10)
    benchmark.schedule()

    combined_csv(args)

# distributed benchmarking
def single_node_scaling(args):
    data = "/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/"
    srun_args = "--mem={mem}G -c {cores} --gres=gpu:{gpus} -t 0 -p {partition}".format(
        mem=12 * args.num_gpu_per_node,
        cores=4 * args.num_gpu_per_node,
        gpus=args.num_gpu_per_node,
        partition=args.partition
    )

    cmd = """srun {srun_args} 
             python -m torch.distributed.launch {distributed_args}
                finetune_on_pregenerated.py \
                --pregenerated_data {data} \
                --bert_model bert-large-uncased \
                --do_lower_case \
                --output_dir finetuned_lm/ \
                --epochs 1 \
                --fp16 \
                --batch_size {batch_size} \
                --seq_length {seq_len} \
                --benchmark \
                --benchmark_dir {benchmark_dir} \
                --benchmark_partition {benchmark_partition}"""

    batch_config = {
        128: [1, 4, 8, 12, 16, 20, 22],
    }

    cmds = []
    increment = -1
    for seq_len, batch_sizes in batch_config.items():
        for x in batch_sizes:
            if not has_ran(args, x, seq_len):
                # increment port num for each iteration
                increment += 1
                # distributed args should be modified upon multi-node training
                distributed_args = "--nproc_per_node {num_gpu_per_node} --nnodes {num_nodes} --node_rank {node_rank} --master_addr {master_addr} --master_port {master_port}".format(
                    num_gpu_per_node=args.num_gpu_per_node,
                    num_nodes=args.num_nodes,
                    node_rank=0,
                    master_addr='localhost',
                    master_port=str(6000 + increment)
                )
                cmds.append(cmd.format(
                    srun_args=srun_args,
                    distributed_args=distributed_args,
                    benchmark_partition=args.partition,
                    batch_size=x,
                    seq_len=seq_len,
                    data=data,
                    benchmark_dir=args.benchmark_dir
                ))

    benchmark = Benchmarking(cmds, max_load=8)
    benchmark.schedule()

    combined_csv(args)


def multi_node_scaling(args):
    assert args.world_size % args.num_gpu_per_node == 0.0
    num_nodes = int(args.world_size / args.num_gpu_per_node)
    assert num_nodes >= 2

    data = "/scratch/hdd001/home/jacoblin/NLP-corpus/wiki_corpus/huggingface/pregen_data_128/"
    srun_args = "--mem={mem}G -c {cores} --gres=gpu:{gpus} -t 0 -p {partition}".format(
        mem=10 * args.num_gpu_per_node,
        cores=4 * args.num_gpu_per_node,
        gpus=args.num_gpu_per_node,
        partition=args.partition
    )

    cmd = """srun {srun_args} 
             sh scripts/benchmark_pretrain_bert_distributed.sh
             --partition={benchmark_partition}
             --seq_len={seq_len}
             --batch_size={batch_size}
             --precision={precision}
             --ngpu_per_node={num_gpu_per_node}
             --nnodes={num_nodes}
             --node_rank={node_rank}
             --master_dir={master_dir}
             --master_output={master_output}"""

    # seq_len : batch size
    batch_config = {
        # 128: [1, 4, 8, 12, 16, 20, 22],
        128: [20],
    }

    cmds = []
    for seq_len, batch_sizes in batch_config.items():
        for batch_size in batch_sizes:
            if not has_ran(args, batch_size, seq_len):
                for rank in range(num_nodes):
                    cmds.append(cmd.format(
                        srun_args=srun_args,
                        benchmark_partition=args.partition,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        precision="fp16",
                        num_gpu_per_node=args.num_gpu_per_node,
                        num_nodes=num_nodes,
                        node_rank=rank,
                        master_dir="master_ip",
                        master_output="{partition}_{nodes}_nodes_{seq_len}_seq_len_{batch_size}_batch_size.ip".format(
                            partition=args.partition,
                            nodes=num_nodes,
                            seq_len=seq_len, 
                            batch_size=batch_size))
                    )

    cmds.reverse()
    benchmark = Benchmarking(cmds, max_load=8)
    benchmark.schedule()

    # combined_csv(args)


def main():
    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser.add_argument('--distributed', action='store_true',
                        help="Multi gpu benchmarking")
    parser.add_argument('--partition', type=str, default='t4',
                        help="Slurm partition")
    parser.add_argument('--world-size', type=int, default=1,
                        help="Number of gpus in total")
    parser.add_argument('--num-gpu-per-node', type=int, default=1,
                        help="Number of gpus in each node ")
    parser.add_argument('--benchmark-dir', type=str, default="benchmark_output",
                        help="Benchmark dir")

    args = parser.parse_args()

    if not os.path.exists(args.benchmark_dir):
        os.mkdir(args.benchmark_dir)
    
    if args.distributed:
        args.num_nodes = int(args.world_size / args.num_gpu_per_node)
        if args.num_nodes == 1:
            single_node_scaling(args)
        else:
            multi_node_scaling(args)
    else:
        baseline(args)


if __name__ == "__main__":
    main()