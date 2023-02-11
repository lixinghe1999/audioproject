import pafy
import os
import subprocess as sp
from tqdm import tqdm
import multiprocessing as mp

audio_codec = 'flac'
audio_container = 'flac'
video_codec = 'h264'
video_container = 'mp4'
duration = 10
def video_download(url, ts_start, video_filepath):
    # Download the video
    video_dl_args = ['ffmpeg', '-y',
                     '-ss', str(ts_start),  # The beginning of the trim window
                     '-i', url,  # Specify the input video URL
                     '-t', str(duration),  # Specify the duration of the output
                     '-f', video_container,  # Specify the format (container) of the video
                     '-r', '1',  # Specify the framerate
                     '-vcodec', 'h264',  # Specify the output encoding
                     video_filepath]
    proc = sp.Popen(video_dl_args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print('fail')
    else:
        print("Downloaded video to " + video_filepath)
def audio_download(url, ts_start, audio_filepath):
    # Download the audio
    audio_dl_args = ['ffmpeg', '-y',
                     '-ss', str(ts_start),  # The beginning of the trim window
                     '-i', url,  # Specify the input video URL
                     '-t', str(duration),  # Specify the duration of the output
                     '-vn',  # Suppress the video stream
                     '-ac', '1',  # Set the number of channels
                     '-sample_fmt', 's16',  # Specify the bit depth
                     '-acodec', audio_codec,  # Specify the output encoding
                     '-ar', '44100',  # Specify the audio sample rate
                     audio_filepath]
    proc = sp.Popen(audio_dl_args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print('fail')
    else:
        print("Downloaded audio to " + audio_filepath)
def down_load(data):
    ytid, ts_start, label = data
    # print("YouTube ID: " + ytid, "Start Time: ({})".format(ts_start))
    # Get the URL to the video page
    video_page_url = 'https://www.youtube.com/watch?v={}'.format(ytid)

    basename_fmt = '{}_{}'.format(ytid, int(ts_start))
    video_filepath = os.path.join('../dataset/VggSound', basename_fmt + '.' + video_container)
    audio_filepath = os.path.join('../dataset/VggSound', basename_fmt + '.' + audio_codec)

    if os.path.isfile(video_filepath) or os.path.isfile(audio_filepath):
        pass
    else:
        try:
            # Get the direct URLs to the videos with best audio and with best video (with audio)
            video = pafy.new(video_page_url)

            best_video = video.getbestvideo()
            best_video_url = best_video.url
            best_audio = video.getbestaudio()
            best_audio_url = best_audio.url

            video_download(best_video_url, ts_start, video_filepath)
            audio_download(best_audio_url, ts_start, audio_filepath)
        except:
            print('fail')



if __name__ == "__main__":
    # Set output settings

    # Load the AudioSet training set
    with open('vggsound.csv') as f:
        lines = f.readlines()

    dl_list = [line.strip().split(',')[:3] for line in lines]
    num_class = dict()
    limit = 200
    dl_list_new = []
    for l in dl_list:
        ytid, ts_start, label = l
        if label in num_class:
            if num_class[label] >= limit:
                continue
            num_class[label] += 1
        else:
            num_class[label] = 1
        dl_list_new.append(l)

    num_processes = os.cpu_count()  # 16
    with mp.Pool(processes=num_processes) as p:
        list(tqdm(p.imap(down_load, dl_list_new), total=len(dl_list_new)))


