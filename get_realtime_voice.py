import pyaudio 
import wave 
import librosa
from pydub import AudioSegment
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#실시간 음성데이터 입력 + 파형 plot 그리기 + 데이터프레임 출력 + 들날숨세트개수

CHUNK = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 2 
#아래의 코드를 따로 실행하고, 본인의 채널 맞춰서 밑에 채널 인수값 바꿔주세요
#p.get_device_info_by_index(1)["name"]
RATE = 44100 
RECORD_SECONDS = 5 
OUTPUT_FILENAME = "realtime_sound"
WAVE_OUTPUT_FILENAME = OUTPUT_FILENAME+'.wav'

p = pyaudio.PyAudio() 

#음성테이터 스트림 열기
stream = p.open(format=FORMAT,
       channels=CHANNELS, 
       rate=RATE, 
       input=True, 
       frames_per_buffer=CHUNK,
       input_device_index=1) 

print("* recording") 

frames = [] 

for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)): 
    data = stream.read(CHUNK) 
    frames.append(data) 

print("* done recording") 

stream.stop_stream() 
stream.close() 
p.terminate() 

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') 
wf.setnchannels(CHANNELS) 
wf.setsampwidth(p.get_sample_size(FORMAT)) 
wf.setframerate(RATE) 
wf.writeframes(b''.join(frames)) 
wf.close() 


#파형 plot 그리기
sound=AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
sound=sound.set_channels(1)
spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, dtype=np.int16)
framerate = spf.getframerate()

Time = np.linspace(0,len(signal)/framerate, num=len(signal))

plt.figure(figsize=(20,5))
plt.title('Signal Wave...')
plt.xticks(np.arange(0,300, step=10))
plt.plot(Time,signal)
plt.grid()
plt.show()

#들숨+날숨세트개수

data_scale = data["value"]/data["value"].max() #데이터값 % 데이터의 최대값 = 정규화
data_scale_mean = data_scale - data_scale.mean().sum() 
data_scale_mean = data_scale_mean/abs(data_scale_mean) 
count=0
for i in range(len(data_scale_mean)-2):
    if data_scale_mean.iloc[i]*data_scale_mean.iloc[i+1] <0:
        if data_scale_mean.iloc[i]*data_scale_mean.iloc[i+2]<0:
            count+=1
    else : 
        continue
print(count//2)

save_df = data.to_csv(OUTPUT_FILENAME + '.csv')