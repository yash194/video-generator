"""
Test script to combine audio and video into one scene.
Usage: python test.py <video_path> <audio_path> <output_path>
"""

from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.fx import Loop
import argparse


def combine_scene(video_path: str, audio_path: str, output_path: str):
    """
    Combine video and audio, looping video if shorter than audio.
    """
    print(f"📹 Loading video: {video_path}")
    video = VideoFileClip(video_path)
    
    print(f"🎵 Loading audio: {audio_path}")
    audio = AudioFileClip(audio_path)
    
    video_duration = video.duration
    audio_duration = audio.duration
    
    print(f"\n⏱️  Video duration: {video_duration:.1f}s")
    print(f"⏱️  Audio duration: {audio_duration:.1f}s")
    
    # Loop video if shorter than audio
    if video_duration < audio_duration:
        print(f"\n🔄 Video is shorter - looping to match audio...")
        video = video.with_effects([Loop(duration=audio_duration)])
    
    # Trim to audio duration
    video = video.subclipped(0, audio_duration)
    
    # Combine
    print(f"\n🔗 Combining audio + video...")
    final = video.with_audio(audio)
    
    print(f"💾 Writing to: {output_path}")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24
    )
    
    # Cleanup
    video.close()
    audio.close()
    final.close()
    
    print(f"\n✅ Done! Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine video and audio into one scene")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("output", help="Path to output file")
    
    args = parser.parse_args()
    
    combine_scene(args.video, args.audio, args.output)
