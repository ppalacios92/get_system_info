# %%
# pip install openpyxl
# pip install cudaq

# %%
import platform
import psutil
import subprocess
import cudaq
import numpy as np
import matplotlib
import sys
from datetime import datetime
import pandas as pd
import os

def get_system_info(computer_name):
    """Collect all system information"""
    
    data = {}
    
    # Computer name
    data['Computer Name'] = computer_name
    
    # Timestamp
    data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # OS - COMPLETE
    data['OS'] = platform.system()
    data['OS Release'] = platform.release()
    data['OS Version'] = platform.version()
    data['Machine'] = platform.machine()
    
    # Linux Distribution - COMPLETE
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = {}
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os_info[key] = value.strip('"')
            
            data['Distribution Name'] = os_info.get('NAME', 'N/A')
            data['Distribution Pretty Name'] = os_info.get('PRETTY_NAME', 'N/A')
            data['Distribution Version'] = os_info.get('VERSION', 'N/A')
            data['Distribution Version ID'] = os_info.get('VERSION_ID', 'N/A')
            data['Distribution ID'] = os_info.get('ID', 'N/A')
    except:
        data['Distribution Name'] = 'N/A'
        data['Distribution Pretty Name'] = 'N/A'
        data['Distribution Version'] = 'N/A'
        data['Distribution Version ID'] = 'N/A'
        data['Distribution ID'] = 'N/A'
    
    # CPU & RAM
    data['Physical CPUs'] = psutil.cpu_count(logical=False)
    data['Logical CPUs'] = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    data['Total RAM (GB)'] = round(mem.total / (1024**3), 2)
    data['Available RAM (GB)'] = round(mem.available / (1024**3), 2)
    data['Used RAM (GB)'] = round(mem.used / (1024**3), 2)
    data['RAM Usage (%)'] = round(mem.percent, 1)
    
    # GPU - COMPLETE with VRAM
    try:
        # Get GPU names and VRAM total
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            data['GPU Count'] = len(gpus)
            
            # Separate GPU names and VRAM
            gpu_names = []
            vram_total = []
            for gpu in gpus:
                parts = gpu.split(',')
                gpu_names.append(parts[0].strip())
                vram_total.append(parts[1].strip())
            
            data['GPU Names'] = ' | '.join(gpu_names)
            data['VRAM Total'] = ' | '.join(vram_total)  # e.g., "24564 MiB | 24564 MiB"
        else:
            data['GPU Count'] = 0
            data['GPU Names'] = 'No GPUs detected'
            data['VRAM Total'] = 'N/A'
    except:
        data['GPU Count'] = 0
        data['GPU Names'] = 'nvidia-smi not available'
        data['VRAM Total'] = 'N/A'
    
    # NVIDIA Driver & CUDA Version
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            data['NVIDIA Driver'] = result.stdout.strip().split('\n')[0]
        else:
            data['NVIDIA Driver'] = 'N/A'
            
        # Get CUDA version from nvidia-smi output
        result_smi = subprocess.run(['nvidia-smi'], 
                                   capture_output=True, text=True, timeout=5)
        if result_smi.returncode == 0:
            # Parse CUDA version from header
            for line in result_smi.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    data['CUDA Version'] = cuda_version
                    break
            else:
                data['CUDA Version'] = 'N/A'
        else:
            data['CUDA Version'] = 'N/A'
    except:
        data['NVIDIA Driver'] = 'N/A'
        data['CUDA Version'] = 'N/A'
    
    # GPU Temperature, Power, Memory Usage, Utilization
    try:
        result = subprocess.run(['nvidia-smi', 
                                '--query-gpu=temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_stats = result.stdout.strip().split('\n')
            temps = []
            powers = []
            vram_used = []
            gpu_utils = []
            for stat in gpu_stats:
                parts = stat.split(',')
                if len(parts) >= 6:
                    temps.append(parts[0].strip())
                    powers.append(f"{parts[1].strip()}/{parts[2].strip()}")
                    vram_used.append(parts[3].strip())
                    gpu_utils.append(parts[5].strip())
            
            data['GPU Temps (C)'] = ' | '.join(temps) if temps else 'N/A'
            data['GPU Power (W)'] = ' | '.join(powers) if powers else 'N/A'
            data['VRAM Used'] = ' | '.join(vram_used) if vram_used else 'N/A'  # e.g., "469 MiB | 18 MiB"
            data['GPU Utilization (%)'] = ' | '.join(gpu_utils) if gpu_utils else 'N/A'
        else:
            data['GPU Temps (C)'] = 'N/A'
            data['GPU Power (W)'] = 'N/A'
            data['VRAM Used'] = 'N/A'
            data['GPU Utilization (%)'] = 'N/A'
    except:
        data['GPU Temps (C)'] = 'N/A'
        data['GPU Power (W)'] = 'N/A'
        data['VRAM Used'] = 'N/A'
        data['GPU Utilization (%)'] = 'N/A'
    
    # Disk
    disk = psutil.disk_usage('/')
    data['Total Disk (GB)'] = round(disk.total / (1024**3), 2)
    data['Used Disk (GB)'] = round(disk.used / (1024**3), 2)
    data['Free Disk (GB)'] = round(disk.free / (1024**3), 2)
    data['Disk Usage (%)'] = round(disk.percent, 1)
    data['Free Disk (%)'] = round(100 - disk.percent, 1)
    
    # Software versions
    data['Python Version'] = platform.python_version()
    data['Python Executable'] = sys.executable
    data['CUDA-Q Version'] = str(cudaq.__version__ if hasattr(cudaq, '__version__') else 'Installed')
    data['NumPy Version'] = np.__version__
    data['Matplotlib Version'] = matplotlib.__version__
    
    return data

def save_to_excel(data, filename='system_history.xlsx'):
    """Save or append data to Excel file"""
    
    df_new = pd.DataFrame([data])
    
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    
    df_final.to_excel(filename, index=False)
    print(f"✓ Data saved to {filename}")
    print(f"✓ Total entries: {len(df_final)}")


# YOU RUN THIS:
computer_name = "esmeralda" 

info = get_system_info(computer_name)

save_to_excel(info)
