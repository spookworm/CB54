import psutil

def kill_processes_by_name(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            pid = proc.info['pid']
            try:
                process = psutil.Process(pid)
                process.terminate()
                print(f"Process with PID {pid} terminated.")
            except psutil.NoSuchProcess:
                print(f"Process with PID {pid} no longer exists.")
