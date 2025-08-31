import whisper
from opencc import OpenCC
import sounddevice as sd
import numpy as np
import threading
import queue

# 创建音频数据队列
audio_queue = queue.Queue()
# 添加停止标志
stop_flag = threading.Event()

def audio_callback(indata, frames, time, status):
    """音频回调函数，实时接收音频数据"""
    if not stop_flag.is_set():
        audio_queue.put(indata.copy())

def realtime_transcribe(model_size="medium"):
    """
    实时录音和转录同时进行
    """
    # 加载模型
    model = whisper.load_model(model_size).to("cuda")
    cc = OpenCC('t2s')
    # 音频参数
    fs = 16000  # 采样率
    chunk_duration = 3  # 每3秒处理一次
    
    print("🎤 开始实时录音和转录...")
    print("按回车键停止")
    print("-" * 40)
    
    # 开始录音流
    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        callback=audio_callback,
        blocksize=int(fs * 0.5)  # 0.5秒的块
    )
    stream.start()
    
    # 音频缓冲区
    audio_buffer = []
    total_duration = 0
    
    try:
        while not stop_flag.is_set():  # 检查停止标志
            # 获取音频数据（非阻塞）
            try:
                audio_data = audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_data)
                total_duration += len(audio_data) / fs
                
                # 每 chunk_duration 秒处理一次
                if total_duration >= chunk_duration:
                    if audio_buffer:
                        # 合并音频数据
                        combined_audio = np.concatenate(audio_buffer, axis=0)
                        audio_float = combined_audio.flatten().astype(np.float32)
                        
                        # 实时转录
                        result = model.transcribe(audio_float, language="zh")
                        text = cc.convert(result["text"])
                        
                        if text.strip():  # 只显示有内容的转录结果
                            print(f"实时转录: {text}")
                            
                        # 写入文件
                        with open("a.txt", 'a', encoding='utf-8') as f:
                            f.write(text + "\n")
                        
                        # 重置缓冲区和计时
                        audio_buffer = []
                        total_duration = 0
                        
            except queue.Empty:
                # 队列为空，继续循环
                continue
                
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        stream.stop()
        stream.close()
        print("\n录音已停止")

# 使用示例
if __name__ == "__main__":
    # 在后台启动实时转录
    transcribe_thread = threading.Thread(target=realtime_transcribe, args=("medium",))
    transcribe_thread.start()
    
    # 主线程等待回车键
    input()  # 等待回车键
    
    # 设置停止标志
    stop_flag.set()
    print("正在停止程序...")
    
    # 等待转录线程结束
    transcribe_thread.join(timeout=2)
    print("程序已完全停止")