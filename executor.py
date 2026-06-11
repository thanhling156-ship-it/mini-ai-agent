import os
import time
import subprocess

QUEUE_FILE = "command_queue.txt"
RESULT_FILE = "command_result.txt"

def run_executor():
    while True:
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, "r") as f:
                    command = f.read().strip()
                
                # Thực thi lệnh
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                
                with open(RESULT_FILE, "w") as f:
                    f.write(result.stdout if result.stdout else result.stderr)
                
                os.remove(QUEUE_FILE)
            except Exception as e:
                with open(RESULT_FILE, "w") as f:
                    f.write(f"Error: {str(e)}")
                if os.path.exists(QUEUE_FILE): os.remove(QUEUE_FILE)
        
        time.sleep(0.5)

if __name__ == "__main__":
    run_executor()