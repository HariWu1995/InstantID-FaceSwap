import subprocess
import psutil

import torch


def get_cpu_info():
    return {
        'CPU_usage': psutil.cpu_percent(),
        'RAM_usage': dict(psutil.virtual_memory()._asdict()),
    }


def get_gpu_memory():

    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_use_info = output_to_list(subprocess.check_output(cmd.split(), 
                                stderr = subprocess.STDOUT))[1:]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}")

    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    return memory_use_values


def get_gpu_profile():
    gpu_profile = []
    for d in range(torch.cuda.device_count()):
        gpu_profile.append({
            'gpu_id': d,
            'memory_allocated_Gb' : round(torch.cuda.memory_allocated(d) / (1024**3), 1),
            'memory_cached_Gb'    : round(torch.cuda.memory_reserved(d) / (1024**3), 1),
        })
    return gpu_profile

