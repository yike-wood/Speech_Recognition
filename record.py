import pyaudio
import wave
import numpy as np
import endpointing
import keyboard
import argparse
import time
import queue


def main(args):
    chunk = 1024  # Each chunk will consist of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Number of audio channels
    fs = 16000  # Record at 16000 samples per second
    filename = args.name + ".wav"
    q = queue.Queue()

    def callback(in_data, frame_count, time_info, status):
        # frames.append(in_data)
        q.put(in_data)
        """ print("callback", q.qsize()) """
        return b"", pyaudio.paContinue

    audio = pyaudio.PyAudio()
    print('Press "space" to start recording.')
    keyboard.wait('space')
    print('-----Now Recording-----')

    stream = audio.open(
        format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True, stream_callback=callback,
    )

    stream.start_stream()
    frame_count = 0
    skip_frame = 16
    background_data = np.empty(0)
    background = 0
    level = 0
    IsSpeech = [1 for i in range(5)]
    onSpeechFlag = False
    loop = True
    frames = []
    start_idx = 0

    while loop:
        data = q.get(block=True)
        rt_data = np.frombuffer(data, np.dtype('<i2'))
        # skip the first 16 frames
        frames.append(data)
        if frame_count < skip_frame:
            frame_count += 1
            continue

        # record first 10 frames as backgound
        if frame_count in range(skip_frame, skip_frame + 10):
            background_data = np.hstack((background_data, rt_data))
            # calculate the background and initial value of level
            if frame_count == skip_frame + 9:
                _, bg_energy_log = endpointing.energy(background_data, chunk)
                background = endpointing.part_sum(0, 10, bg_energy_log) / 10
                level = bg_energy_log[0]
            frame_count += 1
            continue
        # adaptive endpointing detection
        loop, level, background, onSpeechFlag, IsSpeech = endpointDetection(
            rt_data, chunk, level, background, onSpeechFlag, IsSpeech
        )

        frame_count += 1
        if not onSpeechFlag:
            start_idx = frame_count

    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("start index: ", start_idx)
    print("stop index: ", frame_count)
    print("frames length: ", len(frames))

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames[start_idx - 3 : -3]))


def endpointDetection(rt_data, chunk, level, background, onSpeechFlag, IsSpeech):
    terminate, level, background, onSpeechFlag = endpointing.check_stop(rt_data, chunk, level, background, onSpeechFlag)
    IsSpeech.append(terminate)
    check = 0
    # stream terminates with 5 sequential frames not speech
    for idx in range(len(IsSpeech) - 5, len(IsSpeech)):
        check += IsSpeech[idx]
    if check == 0:
        loop = False
    else:
        loop = True
    # check the on-speech flag
    if not onSpeechFlag:
        loop = True

    return loop, level, background, onSpeechFlag, IsSpeech


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", default="tmp")
    args = parser.parse_args()
    main(args)
