from utils import getSubs, get_transcript_vtt, extract_and_save_frames_and_metadata
from os import path as osp
from pathlib import Path
import os
from moviepy.editor import VideoFileClip
import whisper
# output paths to save extracted frames and their metadata 
vid1_dir="Data/"
vid1_filepath="Data/Harsha Bhogle on RCB IPL Retentions @royalchallengersbengaluruYT.mp4"
vid1_transcript_filepath="Data/captions.vtt"
extracted_frames_path = osp.join(vid1_dir, 'extracted_frame')
metadatas_path = vid1_dir

# create these output folders if not existing
Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
Path(metadatas_path).mkdir(parents=True, exist_ok=True)

# call the function to extract frames and metadatas
metadatas = extract_and_save_frames_and_metadata(
                vid1_filepath, 
                vid1_transcript_filepath,
                extracted_frames_path,
                metadatas_path,
            )

path_to_video_no_transcript = vid1_filepath

# declare where to save .mp3 audio
path_to_extracted_audio_file = os.path.join(vid1_dir, 'audio.mp3')

# extract mp3 audio file from mp4 video video file
clip = VideoFileClip(path_to_video_no_transcript)
clip.audio.write_audiofile(path_to_extracted_audio_file)

model = whisper.load_model("small")
options = dict(task="translate", best_of=1, language='en')
results = model.transcribe(path_to_extracted_audio_file, **options)

vtt = getSubs(results["segments"], "vtt")

# path to save generated transcript of video1
path_to_generated_trans = osp.join(vid1_dir, 'generated_videoRCB.vtt')
# write transcription to file
with open(path_to_generated_trans, 'w') as f:
    f.write(vtt)