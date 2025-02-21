import os
import time
import torch
import torch.distributed as dist
import gc
import tracemalloc
import platform
import subprocess
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image

def get_process_memory():
    if platform.system() == "Linux":
        with open('/proc/self/status') as f:
            for line in f:
                if 'VmRSS' in line:
                    return int(line.split()[1]) / 1024
    else:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def get_cpu_usage():
    if platform.system() == "Linux" or platform.system() == "Darwin":
        with open('/proc/stat', 'r') as f:
            lines = f.readlines()
            cpu_line = lines[0].split()
            total_time = sum(map(int, cpu_line[1:]))
            idle_time = int(cpu_line[4])
            return 100 * (1 - (idle_time / total_time))
    elif platform.system() == "Windows":
        try:
            output = subprocess.check_output("wmic cpu get loadpercentage", shell=True)
            return float(output.split()[1])
        except Exception:
            return 0.0
    return 0.0

def log_performance_metrics(start_time, start_memory, start_cpu):
    end_time = time.time()
    end_memory = get_process_memory()
    end_cpu = get_cpu_usage()
    current, peak = tracemalloc.get_traced_memory()
    
    print("\n=== Performance Metrics ===")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Usage: {end_memory - start_memory:.2f} MB")
    print(f"Peak Memory During Execution: {peak / 1024 / 1024:.2f} MB")
    print(f"CPU Utilization: {end_cpu:.2f}%")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    print("========================\n")

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized")

def cleanup():
    dist.destroy_process_group()

def classify_images(image_paths, processor, model, batch_size=8):
    device = torch.device("cuda")
    images = [Image.open(image_path) for image_path in image_paths]
    
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_labels = logits.argmax(-1).tolist()
    return [model.module.config.id2label[label] for label in predicted_labels]

def main(rank):

    gc.collect()
    tracemalloc.start()
    start_memory = get_process_memory()
    start_time = time.time()
    start_cpu = get_cpu_usage()
    
    world_size = torch.cuda.device_count()
    setup_distributed(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model = model.to(device)
    model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))

    if rank == 0:
        print(f"Using {world_size} GPUs with FSDP")

    directory = "pictures"
    batch_size = 8
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                  if os.path.isfile(os.path.join(directory, f))]

    chunk_size = len(image_paths) // world_size
    local_image_paths = image_paths[rank * chunk_size : (rank + 1) * chunk_size]

    for i in range(0, len(local_image_paths), batch_size):
        batch = local_image_paths[i:i+batch_size]
        predictions = classify_images(batch, processor, model, batch_size)
        for img, pred in zip(batch, predictions):
            print(f"GPU {rank} - {img}: Predicted class - {pred}")

    cleanup()
    
    tracemalloc.stop()
    log_performance_metrics(start_time, start_memory, start_cpu)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())