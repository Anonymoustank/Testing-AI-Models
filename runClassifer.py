import time
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
# from datasets import load_dataset
from PIL import Image
import tracemalloc
import gc
import os
import platform

def get_process_memory():
    """Get memory usage in MB using OS-specific commands"""
    if platform.system() == "Linux":
        with open('/proc/self/status') as f:
            for line in f:
                if 'VmRSS' in line:
                    return int(line.split()[1]) / 1024  # Convert KB to MB
    else:
        #shouldn't be needed because I'm using WSL on local and Linux in ICE but still
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def get_cpu_usage():
    """Get CPU utilization as a percentage without psutil."""
    if platform.system() == "Linux" or platform.system() == "Darwin":  # macOS/Linux
        with open('/proc/stat', 'r') as f:
            lines = f.readlines()
            cpu_line = lines[0].split()
            total_time = sum(map(int, cpu_line[1:]))  # Sum of all time components
            idle_time = int(cpu_line[4])  # Idle time
            return 100 * (1 - (idle_time / total_time))  # CPU usage percentage

    elif platform.system() == "Windows":  # Windows method using WMIC
        try:
            output = subprocess.check_output("wmic cpu get loadpercentage", shell=True)
            return float(output.split()[1])
        except Exception:
            return 0.0  # Fallback if command fails

    return 0.0  # Default if OS is unsupported

def track_performance(func):
    def wrapper(*args, **kwargs):
        gc.collect()  # Request garbage collection
        
        tracemalloc.start()  # Start memory tracking
        start_memory = get_process_memory()
        start_time = time.time()
        start_cpu = get_cpu_usage()  # Capture initial CPU usage

        result = func(*args, **kwargs)  # Execute function
        
        end_time = time.time()  # Capture end time
        end_memory = get_process_memory()
        end_cpu = get_cpu_usage()  # Capture final CPU usage
        current, peak = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()  # Stop memory tracking

        print("\n=== Performance Metrics ===")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Memory Usage: {end_memory - start_memory:.2f} MB")
        print(f"Peak Memory During Execution: {peak / 1024 / 1024:.2f} MB")
        print(f"CPU Utilization: {end_cpu:.2f}%")
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        print("========================\n")

        return result
    return wrapper

def classify_image(image_path):
    # https://huggingface.co/docs/datasets/en/loading

    # dataset = load_dataset("huggingface/cats-image")
    # imagePath = dataset["test"]["image"][0]  # Get the first image from the test set
    # print(imagePath)
    # pil_image = Image.open(imagePath)
    if isinstance(image_path, Image.Image):
        pil_image = image_path
    else:
        pil_image = Image.open(image_path)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = processor(pil_image, return_tensors="pt") # image processing needed for model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    
    with torch.no_grad():
        logits = model(**inputs).logits # prediction for what image is done by model

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label] # return the label of the image

@track_performance
def main():
    directory = "pictures"
    iterations = 0
    maxIterations = 100
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            iterations += 1
            image_path = f
            prediction = classify_image(image_path)
            print(f"Predicted class: {prediction}")
        if iterations > maxIterations:
            break

main()
