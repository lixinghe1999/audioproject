
if __name__ == "__main__":
    # Set output settings
    audio_codec = 'flac'
    audio_container = 'flac'
    video_codec = 'h264'
    video_container = 'mp4'
    # Load the AudioSet training set
    with open('vggsound.csv') as f:
        lines = f.readlines()

    dl_list = [line.strip().split(',')[:3] for line in lines[3:]]