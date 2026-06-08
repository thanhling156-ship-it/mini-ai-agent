# Sao chép toàn bộ đoạn mã này và lưu thành file run_agent.py trong cùng thư mục với ./gpt_agent_output

import os
import sys
import subprocess
import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast, StoppingCriteria, StoppingCriteriaList

# ==========================================
# 1. INITIALIZATION & ARTIFACT LOADING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT_OUTPUT_DIR = "./gpt_agent_output"

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
def execute_system_command(action_text):
    """
    Phân tích chuỗi hành động của Agent và gọi lệnh tương tác với Hệ điều hành thực tế.
    Trả về 'success' hoặc 'failed' làm dữ liệu phản hồi (Feedback) cho Agent.
    """
    print(f"\n[OS Executor] Đang xử lý lệnh: \"{action_text}\"")
    action_text = action_text.lower().strip()
    
    try:
        # Nhóm hành động: MỞ ỨNG DỤNG (OPEN / LAUNCH)
        if "open" in action_text or "launch" in action_text or "run" in action_text:
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
                    subprocess.Popen(["taskmgr"])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", "Activity Monitor"])
                else:
                    subprocess.Popen(["gnome-system-monitor"])
                return "success"

        # Nhóm hành động: ĐÓNG ỨNG DỤNG (CLOSE / KILL / TERMINATE)
        elif "close" in action_text or "kill" in action_text or "terminate" in action_text:
    
            if "task manager" in action_text:
                # Do toàn bộ file đã có quyền Admin từ đầu, ta có thể bỏ qua bước check chữ "admin" trong text
                # Hoặc nếu muốn giữ đúng logic ReAct tuần tự thì check biến trạng thái:
                print("[OS Executor] Đang thực thi lệnh đóng Task Manager với đặc quyền Quản trị...")
                if sys.platform == "win32":
                    # Lệnh taskkill thực tế sẽ chạy thành công 100% vì python đã có quyền Admin
                    result = subprocess.run(["taskkill", "/f", "/im", "taskmgr.exe"], capture_output=True, text=True)
                    if "SUCCESS" in result.stdout.upper() or result.returncode == 0:
                        return "success"
                    else:
                        print(f"[OS Executor] Lỗi từ hệ thống: {result.stderr}")
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

        # Nhóm hành động: LEO THANG ĐẶC QUYỀN (ADMIN PRIVILEGE ELEVATION)
        elif "admin privilege" in action_text:
            # Vì file đã chạy bằng quyền Admin thật từ lúc mở, bước này đóng vai trò xác nhận logic cho Agent biết
            print("[OS Executor] Xác thực: Trạng thái Quản trị viên hệ thống đã sẵn sàng.")
            return "success"
        

        print("[OS Executor] Lệnh hệ thống không xác định hoặc chưa được hỗ trợ.")
        return "failed"

    except Exception as error:
        print(f"[OS Executor] Lỗi ngoại lệ xảy ra khi thực thi lệnh: {error}")
        return "failed"

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
                    raw_step_command = new_tokens_str.replace("[ACT]", "").strip()
                    
                    # Truyền ngữ cảnh quyền admin cho executor nếu đang ở bước khôi phục lỗi
                    if "step 2" in raw_step_command and "admin" in raw_step_command:
                        action_payload = f"{raw_step_command} for task manager"
                    else:
                        action_payload = raw_step_command
                        
                    execution_result = execute_system_command(action_payload)
                    
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
    # KHỐI KIỂM TRA VÀ TỰ ĐỘNG XIN QUYỀN ADMIN CHO TOÀN BỘ FILE CODE
    if sys.platform == "win32":
        import ctypes
        def is_admin():
            try:
                return ctypes.windll.shell32.IsUserAnAdmin()
            except:
                return False

        # Nếu file CHƯA được chạy bằng quyền Admin
        if not is_admin():
            print("[Hệ Thống] Kích hoạt hộp thoại UAC để cấp quyền Quản trị cho Agent...")
            # Gọi lại chính file run_agent.py này nhưng với đặc quyền tối cao (runas)
            # sys.executable là đường dẫn tới python.exe, sys.argv là các tham số truyền vào
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            sys.exit(0)  # Tắt file quyền thường hiện tại đi, nhường chỗ cho file quyền Admin vừa mở

    # Khi đã có quyền Admin (hoặc chạy trên hệ điều hành khác), tiến hành chạy vòng lặp chính
    main()