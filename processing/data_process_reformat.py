# This file doesn't generate grid images or mp3 audio files. It's only used to reformat the jsonl file. 

import pandas as pd
import json
import cv2
from PIL import Image
import numpy as np

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from moviepy.editor import AudioFileClip, VideoFileClip

def sample_frames(video_path, num_frames=4):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to sample
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def create_grid(frames, grid_size=(2, 2)):
    assert len(frames) == grid_size[0] * grid_size[1]

    # assert , "Number of frames does not match grid size"
    
    # Get dimensions of the images
    h, w, _ = frames[0].shape
    
    # Create an empty canvas for the grid
    grid_img = Image.new('RGB', (w * grid_size[1], h * grid_size[0]))
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        x = (i % grid_size[1]) * w
        y = (i // grid_size[1]) * h
        grid_img.paste(img, (x, y))
    
    return grid_img


def extract_audio_from_video(video_path, audio_output_path):

    # # Load the video file
    video = VideoFileClip(video_path)
    
    # # Extract the audio
    audio = video.audio
    
    # # Write the audio to an mp3 file
    audio.write_audiofile(audio_output_path)


# def decode_unicode_escapes(text):
#     if isinstance(text, str):
#         # replace \u2019 with '
#         text = text.replace("\u2019", "'")
#         text = text.replace("\u2026", "...")
#         text = text.replace("\u2014", "——")
#         return text.encode('latin1').decode('unicode_escape')
#     return text

if __name__ == "__main__":

    # load the .csv file, specify the appropriate path
    df_train = pd.read_csv('/share/pi/schul/jennyxu/MELD.Raw/train_sent_emo.csv') 

    # video_folder
    video_folder = "/share/pi/schul/jennyxu/MELD.Raw/train_splits/"

    # Path to the saved data 
    data_processed_path = "data_processed/"

    # Path to the jsonl file 
    jsonl_file = data_processed_path + "annotations_with_transcript.jsonl"

    index = 0

    with open(jsonl_file, 'w') as file:
        for index, row in df_train.iterrows():

            record = {
                "id": index,
            }

            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            video_name = "dia{}_utt{}.mp4".format(dialogue_id, utterance_id)
            video_path = video_folder + video_name

            ####################### Logger 
            print(index, video_name)

            # Sample 4 frames uniformly from the video and make a 2x2 grid of images.
            # frames = sample_frames(video_path)
            # if (len(frames) != 4):
            #     print("Number of frames isn't 4")
            #     continue
            # grid_img = create_grid(frames)

            # save image to the images folder. 
            # grid_img.save(data_processed_path + "images/{}.png".format(index))
            record["image"] = "images/{}.png".format(index)
            
            # Obtain the audio file. 
            audio_path = "audios/{}.mp3".format(index)
            # audio_output_path = data_processed_path + audio_path
            # extract_audio_from_video(video_path, audio_output_path)
            record["audio"] = audio_path

            transcript = row['Utterance']

            # construct the conversation and Extract the emotion 
            record["conversations"] = [
                {
                    "from": "human",
                    "value": "<image>\n<audio>\nAbove are 4 frames and an audio clip from a video. The speaker said, '{}' What is the emotion of the speaker in this video?".format(transcript)
                },
                {
                    "from": "gpt",
                    "value": row["Emotion"]
                }
            ]

            # Add in a field to indicate the name of the original video
            # record['original_video'] = video_name

            

            # Add in the text/caption
            # record['utterance'] = row['Utterance']


            file.write(json.dumps(record) + '\n')


