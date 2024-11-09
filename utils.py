from PIL import Image
import json
import os
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
from io import StringIO, BytesIO
import base64
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import os
from os import path as osp
import json
import webvtt
from moviepy.editor import VideoFileClip
import base64
from pytubefix import YouTube, Stream
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
import glob
import textwrap
from typing import Iterator, TextIO
from lvlm import lvlm_inference_module

# encoding image at given path or PIL Image using base64
def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, Image.Image):
        return image_path_or_PIL_img
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            image_data = image_file.read()
            image = Image.open(BytesIO(image_data))
            return image



def get_clip_embeddings(text=None, image=None):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    if text and image:
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True, max_length=77)
        outputs = model(**inputs)
        return outputs.text_embeds[0].tolist()
    elif text:
        inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True, max_length=77)
        outputs = model.get_text_features(**inputs)
        return outputs[0].tolist()
    elif image:
        inputs = processor(images=image, return_tensors="pt",truncation=True, max_length=77)
        outputs = model.get_image_features(**inputs)
        return outputs[0].tolist()

def download_video(video_url, path='./tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    
    yt = YouTube(video_url, on_progress_callback=progress_callback,use_po_token=True,allow_oauth_cache=False)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
    if stream is None:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, stream.default_filename)
    if not os.path.exists(filepath):   
        print('Downloading video from YouTube...')
        pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
        stream.download(path)
        pbar.close()
    return filepath

def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]

    return video_url
    
# if this has transcript then download
def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

    return filepath

def str2time(strtime):
    strtime=strtime.strip('"')
    hrs,mins,seconds = [float(c) for c in strtime.split(":")]
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

# function `extract_and_save_frames_and_metadata``:
#   receives as input a video and its transcript
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata(
        path_to_video, 
        path_to_transcript, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    trans = webvtt.read(path_to_transcript)
    
    # iterate transcript file
    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        # get the start time and end time in seconds
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        # get the time in ms exactly 
        # in the middle of start time and end time
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", ' ')
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            )
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = osp.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )


def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# function extract_and_save_frames_and_metadata_with_fps
#   receives as input a video 
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata_with_fps(
        path_to_video,  
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # Get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    curr_frame = 0
    idx = -1
    while(True):
        # iterate all frames
        ret, frame = video.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            idx = idx + 1
        
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames, 
                            img_fname
                        )
            cv2.imwrite(img_fpath, image)

            # generate caption using lvlm_inference
            b64_image = encode_image(img_fpath)
            result={"prompt":"Describe the image","image":b64_image}
            caption = lvlm_inference_module(result)
                
            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1
        
    # save metadata of all extracted frames
    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json')
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_retrieved_results(results):
    print(f'There is/are {len(results)} retrieved result(s)')
    print()
    for i, res in enumerate(results):
        print(f'The caption of the {str(i+1)}-th retrieved result is:\n"{results[i].page_content}"')
        print()
        display(Image.open(results[i].metadata['extracted_frame_path']))
        print("------------------------------------------------------------")

