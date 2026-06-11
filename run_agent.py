# Sao chép toàn bộ đoạn mã này và lưu thành file run_agent.py trong cùng thư mục với ./gpt_agent_output

import os
import sys
import subprocess
import torch
import ctypes
import win32file
import os
import subprocess
import time
import psutil
import time
from transformers import GPT2LMHeadModel, BertTokenizerFast, StoppingCriteria, StoppingCriteriaList

# ==========================================
# 1. INITIALIZATION & ARTIFACT LOADING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT_OUTPUT_DIR = "./gpt_agent_output"
QUEUE_FILE = "command_queue.txt"
RESULT_FILE = "command_result.txt"

if not os.path.exists(GPT_OUTPUT_DIR):
    print(f"Error: Directory '{GPT_OUTPUT_DIR}' not found. Please train the model first.")
    sys.exit(1)

tokenizer = BertTokenizerFast.from_pretrained(GPT_OUTPUT_DIR)
model_gpt = GPT2LMHeadModel.from_pretrained(GPT_OUTPUT_DIR).to(DEVICE)
model_gpt.eval()

# Custom stopping criteria to pause generation exactly at [ACT] token
class StopAtTokenCriteria(StoppingCriteria):
    def __init__(self, target_token_id):
        self.target_token_id = target_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.target_token_id

act_token_id = tokenizer.convert_tokens_to_ids("[ACT]")
end_token_id = tokenizer.convert_tokens_to_ids("[END]")

# ==========================================
# 2. REAL OS ENVIRONMENT EXECUTION (EXTERNAL ACTIONS)
# ==========================================

def is_executor_running():
    # Kiểm tra xem có tiến trình nào tên python chạy executor.py không
    for proc in psutil.process_iter(['name', 'cmdline']):
        if proc.info['cmdline'] and 'executor.py' in proc.info['cmdline']:
            return True
    return False

def ensure_admin_executor():
    if not is_executor_running():
        print("Đang khởi động Executor với quyền Admin...")
        # Dùng PowerShell gọi RunAs để kích hoạt UAC
        cmd = "Start-Process python -ArgumentList 'executor.py' -Verb RunAs"
        subprocess.run(["powershell", "-Command", cmd])
        time.sleep(2) # Chờ executor khởi động

def send_admin_command(command):
    if not is_executor_running():
        return "failed"

    with open(QUEUE_FILE, "w") as f:
        f.write(command)
    
    # Chờ kết quả
    timeout = 10
    start_time = time.time()
    while not os.path.exists(RESULT_FILE):
        if time.time() - start_time > timeout:
            return "Lỗi: Timeout khi đợi Admin phản hồi."
        time.sleep(0.2)
        
    with open(RESULT_FILE, "r") as f:
        res = f.read()
    os.remove(RESULT_FILE)
    return res

def execute_system_command(action_text):
    """
    Phân tích chuỗi hành động của Agent và gọi lệnh tương tác với Hệ điều hành thực tế.
    Trả về 'success' hoặc 'failed' làm dữ liệu phản hồi (Feedback) cho Agent.
    """
    global IS_ADMIN
    print(f"\n[OS Executor] Đang xử lý lệnh: \"{action_text}\"")
    action_text = action_text.lower().strip()

    # Xử lý các lệnh cụ thể dựa trên từ khóa chính
    try:
        # Nhóm Đóng ứng dụng (Kill)
        if "close" in action_text or "kill" in action_text or "terminate" in action_text:
            if "task manager" in action_text:
                print("[OS Executor] Đang thực thi lệnh đóng Task Manager với đặc quyền Quản trị...")
                if sys.platform == "win32":
                    result = send_admin_command("taskkill /f /im taskmgr.exe")
                    if "has been terminated" in result.lower() or "success" in result.lower():
                        return "success"
                    else:
                        print(f"[OS Executor] Lệnh Admin thất bại ")
                        return "failed"
                else:
                    subprocess.run(["pkill", "-f", "taskmgr"], capture_output=True)
                    return "success"
                
            # 2. Xử lý đóng Google Chrome (Mới cập nhật)
            elif "chrome" in action_text:
                print("[OS Executor] Đang đóng tiến trình Google Chrome...")
                if sys.platform == "win32":
                    # Khóa /f bắt buộc đóng, /im chỉ định tên image tiến trình
                    subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], capture_output=True)
                elif sys.platform == "darwin": # macOS
                    subprocess.run(["pkill", "-f", "Google Chrome"], capture_output=True)
                else: # Linux
                    subprocess.run(["pkill", "-f", "chrome"], capture_output=True)
                return "success"
                
            # 3. Xử lý đóng Terminal / Command Prompt (Mới cập nhật)
            elif "terminal" in action_text or "cmd" in action_text:
                print("[OS Executor] Đang xử lý đóng Terminal/Command Prompt...")
                if sys.platform == "win32":
                    # Đóng các cửa sổ cmd thông thường (tránh tắt nhầm chính cửa sổ Admin đang chạy Agent)
                    subprocess.run(["taskkill", "/f", "/im", "cmd.exe"], capture_output=True)
                elif sys.platform == "darwin":
                    subprocess.run(["pkill", "-f", "Terminal"], capture_output=True)
                else:
                    subprocess.run(["pkill", "-f", "x-terminal-emulator"], capture_output=True)
                return "success"
        
        # Nhóm Mở ứng dụng (Open)
        elif any(keyword in action_text for keyword in ["open", "launch", "run"]):
            if "chrome" in action_text:
                print("[OS Executor] Đang mở Google Chrome...")
                if sys.platform == "win32":
                    subprocess.Popen(["start", "chrome"], shell=True)
                elif sys.platform == "darwin": # macOS
                    subprocess.Popen(["open", "-a", "Google Chrome"])
                else: # Linux
                    subprocess.Popen(["google-chrome"])
                return "success"
                
            elif "terminal" in action_text or "cmd" in action_text:
                print("[OS Executor] Đang mở Terminal/Command Prompt...")
                if sys.platform == "win32":
                    subprocess.Popen(["start", "cmd"], shell=True)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", "Terminal"])
                else:
                    subprocess.Popen(["x-terminal-emulator"])
                return "success"
                
            elif "task manager" in action_text:
                print("[OS Executor] Đang mở Task Manager...")
                if sys.platform == "win32":
                    subprocess.Popen(["taskmgr"], shell=True)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", "Activity Monitor"])
                else:
                    subprocess.Popen(["gnome-system-monitor"])
                return "success"
            return "success"
        
        elif "admin privilege" in action_text:
            print("[OS Executor] Phát hiện yêu cầu quyền Admin...")
            ensure_admin_executor()
            return "success"
            
        return "failed"
    except Exception as e:
        return f"failed: {e}"

# ==========================================
# 3. INTERACTIVE REACT VISUALIZATION LOOP
# ==========================================
def main():
    print("="*60)
    print("         GPT RE-ACT SYSTEM AUTOMATION INTERACTIVE AGENT       ")
    print("="*60)
    print("Nhập câu lệnh của bạn bằng tiếng Anh (Ví dụ: 'open chrome please')")
    print("Gõ 'exit' hoặc 'quit' để dừng chương trình.\n")
    
    while True:
        try:
            user_input = input("User Request >>> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Đang đóng Agent. Tạm biệt.")
                break
                
            # Khởi tạo chuỗi quỹ đạo (Trajectory) ban đầu
            current_trajectory = f"[CLS] {user_input} [SEP]"
            step_count = 0
            max_steps = 5
            is_completed = False
            
            print(f"\n--- [Agent Tiến Hành Xử Lý Từ Ngữ Cảnh] ---")
            
            while step_count < max_steps and not is_completed:
                input_ids = tokenizer.encode(current_trajectory, add_special_tokens=False, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = model_gpt.generate(
                        input_ids,
                        max_new_tokens=20,
                        pad_token_id=tokenizer.pad_token_id,
                        stopping_criteria=StoppingCriteriaList([StopAtTokenCriteria(act_token_id)]),  # Tạo mới mỗi lần
                        eos_token_id=end_token_id,
                        do_sample=False,
                        repetition_penalty=1.3,  # Tránh lặp token
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                if generated_text == current_trajectory:
                    print("[Hệ Thống] Chuỗi quỹ đạo bị đình trệ (Stagnated). Dừng tiến trình.")
                    break
                
                # Decode chỉ phần token mới sinh ra (tránh lệch do string slicing)
                new_token_ids = outputs[0][input_ids.shape[1]:]
                new_tokens_str = tokenizer.decode(new_token_ids, skip_special_tokens=False).strip()
                current_trajectory = generated_text
                
                print(f"Agent Step {step_count + 1}: {new_tokens_str}")
                
                # Chỉ check [END], không dùng "done" để tránh match nhầm
                if "[END]" in current_trajectory:
                    is_completed = True
                    print("---------------------------------------")
                    print("[Trạng Thái] Hoàn thành quỹ đạo thực thi lệnh thành công.")
                    break
                
                # Nếu Agent phát hành động dạng [ACT], bóc tách text mang đi gọi lệnh OS thực tế
                if current_trajectory.endswith("[ACT]"):
                    # Chỉ cần lấy command từ Agent và gọi hàm xử lý
                    raw_step_command = new_tokens_str.replace("[ACT]", "").strip()

                    # Loại bỏ if-else rườm rà tại đây
                    execution_result = execute_system_command(raw_step_command)
                    
                    # Nối phản hồi từ môi trường (Feedback) vào lịch sử để Agent đọc tiếp lượt sau
                    feedback_str = f" {execution_result} [SEP]"
                    current_trajectory += feedback_str
                    print(f"Phản Hồi Môi Trường (OS): {execution_result.upper()}")
                    
                step_count += 1
            
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nĐang tắt Agent bằng lệnh ngắt tổ hợp phím.")
            break
        except Exception as e:
            print(f"\n[Lỗi Runtime] Đã có lỗi xảy ra: {e}\n")

if __name__ == "__main__":
    
    main()
# Yêu cầu: khi đã có admin thì có thể thực hiện luôn lệnh mà không cần gọi UAC nữa
# Vấn đề: Agent khi kill task manager luôn đi theo 1 chuỗi tuyến tính mà không thể phân biệt được đã có quyền admin hay chưa
# => Dẫn đến bị thừa bước gọi UAC vì lệnh ban đầu luôn lỗi bất kể có admin hay không
# 
# Nhưng nếu kiểm tra quyền admin thì sẽ thực hiện luôn lệnh
# => Dẫn đến chuỗi ReAct bị thu hẹp còn 1 bước duy nhất là "close task manager" => Agent không có cơ hội sinh bước gọi UAC 

# Vậy mình nên để agent sinh ra như này vì lệnh "close task manager" ở terminal thường luôn trả về lỗi
# => Agent luôn có bước kiểm tra lại "Admin Privilege" 

# Thì ở command khác lệnh sẽ lúc được lúc không, còn hiện tại thì luôn luôn fail => Agent luôn có cơ hội sinh ra bước "Admin Privilege" để tự nâng quyền nếu cần thiết
# Vậy nên sẽ sử dụng tạm LBYL mang tính đóng khung mặc dù k được tự nhiên nhưng sẽ đảm bảo k bị thừa step 2

# Tức là chúng ta cần tính đồng nhất chứ k phải là step 1 luôn fail rồi step 3 lại theo 1 lối khác dù đều thực hiện cùng nhiệm vụ
# => Mang tính tuyến tính