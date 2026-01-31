"""
Video Generation Pipeline

Complete scene-by-scene video generation from screenplay JSON.
Uses state-of-the-art open-source models with GPU memory management for 40GB VRAM.

Models:
- Audio: Kokoro TTS (82M params, natural narration, 24kHz)
- Image: FLUX.1 Dev (best prompt adherence & text rendering)
- Video: LTXVideo (prompt-aware image-to-video, 24fps)

Dependencies:
    pip install torch diffusers transformers accelerate
    pip install kokoro soundfile numpy
    pip install moviepy
    pip install opencv-python pillow
"""

import os
import gc
import json
import torch
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass


# ============================================================================
# GPU MEMORY MANAGER
# ============================================================================

class GPUMemoryManager:
    """
    Manages GPU memory by loading/unloading models sequentially.
    Only one model active at a time to maximize VRAM usage.
    """
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        
    def get_free_vram(self) -> float:
        """Get free VRAM in GB."""
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / (1024**3)
            total = torch.cuda.mem_get_info()[1] / (1024**3)
            print(f"💾 VRAM: {free:.1f}GB free / {total:.1f}GB total")
            return free
        return 0
    
    def unload_current(self):
        """Unload current model and clear GPU memory."""
        if self.current_model is not None:
            print(f"🗑️  Unloading {self.current_model_name}...")
            del self.current_model
            self.current_model = None
            self.current_model_name = None
        
        # Aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def load_model(self, name: str, loader_fn: Callable):
        """
        Load a model, unloading any current model first.
        
        Args:
            name: Model name for logging
            loader_fn: Function that returns the loaded model
        """
        # Unload current model first
        self.unload_current()
        
        print(f"📦 Loading {name}...")
        self.get_free_vram()
        
        self.current_model = loader_fn()
        self.current_model_name = name
        
        self.get_free_vram()
        print(f"✓ {name} loaded")
        
        return self.current_model


# Global memory manager
memory_manager = GPUMemoryManager()


# ============================================================================
# AUDIO GENERATOR (Kokoro TTS - State of the Art Quality)
# ============================================================================

class AudioGenerator:
    """
    Generate high-quality speech using Kokoro TTS.
    Kokoro is an 82M parameter model with quality comparable to much larger systems.
    Produces natural, expressive narration at 24kHz.
    """
    
    def __init__(
        self, 
        voice: str = "af_heart",  # Available: af_heart, af_bella, am_adam, am_michael, bf_emma, bm_george
        speed: float = 1.0
    ):
        """
        Initialize Kokoro TTS.
        
        Voice options:
        - af_heart: American female, warm expressive (default, best for narration)
        - af_bella: American female, professional
        - am_adam: American male, professional
        - am_michael: American male, warm
        - bf_emma: British female
        - bm_george: British male
        """
        self.voice = voice
        self.speed = speed
        self.pipeline = None
    
    def _load(self):
        """Load Kokoro TTS pipeline."""
        from kokoro import KPipeline
        
        # Detect language from voice prefix
        lang = 'a' if self.voice.startswith('a') else 'b'
        
        pipeline = KPipeline(lang_code=lang, device="cuda")
        return pipeline
    
    def load(self):
        """Load model via memory manager."""
        self.pipeline = memory_manager.load_model("Kokoro-TTS", self._load)
    
    def generate(self, text: str, output_path: str) -> str:
        """
        Generate speech audio from text.
        
        Args:
            text: Narration text to speak
            output_path: Path to save audio file (.wav)
            
        Returns:
            Path to generated audio file
        """
        import soundfile as sf
        
        if self.pipeline is None:
            self.load()
        
        # Generate audio
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed
        )
        
        # Collect all audio chunks
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
        
        # Concatenate and save
        import numpy as np
        full_audio = np.concatenate(audio_chunks)
        sf.write(output_path, full_audio, 24000)  # Kokoro outputs at 24kHz
        
        return output_path


# ============================================================================
# IMAGE GENERATOR (Alibaba Z-Image Turbo - #1 Open Source Model)
# ============================================================================

class ImageGenerator:
    """
    Generate high-quality images using Alibaba Z-Image Turbo.
    Ranked #1 open-source model on Text-to-Image leaderboard.
    Features:
    - 6B parameters, distilled for speed
    - Excellent photorealism and text rendering
    - Bilingual (English + Chinese) text support  
    - Only 8 steps needed for high quality
    - Runs on 16GB VRAM
    """
    
    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        width: int = 1024,
        height: int = 576,  # 16:9 aspect for video
        num_inference_steps: int = 28,  # Online demos use 28 steps for best quality
        guidance_scale: float = 4.5  # Online demos use 4.5 guidance
    ):
        self.model_id = model_id
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = None
    
    def _load(self):
        """Load Z-Image Turbo pipeline."""
        from diffusers import DiffusionPipeline
        
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        return pipe
    
    def load(self):
        """Load model via memory manager."""
        self.pipe = memory_manager.load_model("Z-Image-Turbo", self._load)
    
    def generate(
        self, 
        prompt: str, 
        output_path: str,
        negative_prompt: str = None
    ) -> str:
        """
        Generate high-quality image from visual prompt.
        
        Args:
            prompt: Visual description prompt
            output_path: Path to save image (.png)
            negative_prompt: Things to avoid in generation
            
        Returns:
            Path to generated image
        """
        if self.pipe is None:
            self.load()
        
        # Use the same negative prompt as online demos
        if negative_prompt is None:
            negative_prompt = (
                "low quality, worst quality, blurry, distorted, ugly, "
                "bad anatomy, watermark, signature, text, logo, "
                "deformed, disfigured, mutation, mutated, extra limbs, "
                "bad proportions, cropped, out of frame"
            )
        
        # Generate image with optimized settings
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale
            ).images[0]
        
        # Save
        image.save(output_path)
        return output_path


# ============================================================================
# VIDEO GENERATOR (CogVideoX-5B I2V - Image + Text to Video)
# ============================================================================

class VideoGenerator:
    """
    Generate video from image + text prompt using CogVideoX-5B.
    
    Features:
    - Takes BOTH image AND text prompt for generation
    - 5B parameters, high quality video
    - 6 seconds of video at 8fps (49 frames)
    - 720x480 resolution
    - ~16GB VRAM with optimizations
    """
    
    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-5b-I2V",
        num_frames: int = 49,  # 6 seconds at 8fps
        fps: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0
    ):
        self.model_id = model_id
        self.num_frames = num_frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = None
    
    def _load(self):
        """Load CogVideoX I2V pipeline."""
        from diffusers import CogVideoXImageToVideoPipeline
        
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        
        # Enable memory optimizations for 40GB VRAM
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        return pipe
    
    def load(self):
        """Load model via memory manager."""
        self.pipe = memory_manager.load_model("CogVideoX-5B-I2V", self._load)
    
    def generate(
        self,
        image_path: str,
        motion_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate video from image with text-guided motion.
        
        Args:
            image_path: Path to input image (starting frame)
            motion_prompt: Text description of the video motion
            output_path: Path to save video (.mp4)
            
        Returns:
            Path to generated video
        """
        from PIL import Image
        from diffusers.utils import export_to_video
        
        if self.pipe is None:
            self.load()
        
        # Load and resize input image (CogVideoX: 720x480)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((720, 480))
        
        # Create video prompt
        video_prompt = f"{motion_prompt}. Smooth motion, cinematic, high quality."
        
        # Generate video frames with both image and text
        with torch.no_grad():
            output = self.pipe(
                image=image,
                prompt=video_prompt,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                use_dynamic_cfg=True
            )
            frames = output.frames[0]
        
        # Export to video
        export_to_video(frames, output_path, fps=self.fps)
        
        return output_path


# ============================================================================
# SCENE COMBINER (Audio + Video Sync)
# ============================================================================

class SceneCombiner:
    """Combine audio and video, handling duration mismatches."""
    
    def combine(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        match_to: str = "audio"  # "audio" or "video"
    ) -> str:
        """
        Combine video and audio into single clip.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to output file
            match_to: Which duration to match ("audio" loops video, "video" trims audio)
            
        Returns:
            Path to combined video
        """
        from moviepy import VideoFileClip, AudioFileClip
        from moviepy.video.fx import Loop
        
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        video_duration = video.duration
        audio_duration = audio.duration
        
        print(f"  Video: {video_duration:.1f}s, Audio: {audio_duration:.1f}s")
        
        if match_to == "audio":
            # Match video to audio duration
            if video_duration < audio_duration:
                # Loop video to match audio
                loops_needed = int(audio_duration / video_duration) + 1
                video = video.with_effects([Loop(duration=audio_duration)])
            video = video.subclipped(0, audio_duration)
        else:
            # Match audio to video duration
            if audio_duration > video_duration:
                audio = audio.subclipped(0, video_duration)
            # If audio is shorter, video will have silence at end
        
        # Combine
        final = video.with_audio(audio)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            # preset="medium",
            # verbose=False,
            # logger=None
        )
        
        # Cleanup
        video.close()
        audio.close()
        final.close()
        
        return output_path


# ============================================================================
# VIDEO STITCHER (Concatenate Scenes)
# ============================================================================

class VideoStitcher:
    """Stitch multiple scene clips into final video."""
    
    def stitch(
        self,
        scene_paths: list,
        output_path: str,
        transition: str = "none"  # "none", "fade", "crossfade"
    ) -> str:
        """
        Concatenate scene clips into final video.
        
        Args:
            scene_paths: List of scene video paths in order
            output_path: Path to final output video
            transition: Type of transition between scenes
            
        Returns:
            Path to final video
        """
        from moviepy import VideoFileClip, concatenate_videoclips
        from moviepy.video.fx import FadeIn, FadeOut
        
        print(f"🎬 Stitching {len(scene_paths)} scenes...")
        
        clips = []
        for path in scene_paths:
            clip = VideoFileClip(path)
            
            if transition == "fade":
                clip = clip.with_effects([FadeIn(0.5), FadeOut(0.5)])
            
            clips.append(clip)
        
        # Concatenate
        if transition == "crossfade":
            # Crossfade requires overlap
            final = concatenate_videoclips(clips, method="compose", padding=-0.5)
        else:
            final = concatenate_videoclips(clips, method="compose")
        
        # Write final video
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            # preset="medium",
            # verbose=False,
            # logger=None
        )
        
        # Cleanup
        for clip in clips:
            clip.close()
        final.close()
        
        print(f"✓ Final video: {output_path}")
        return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class VideoPipeline:
    """
    Complete video generation pipeline.
    Processes screenplay JSON scene-by-scene.
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        voice: str = "am_adam",  # Kokoro voice: am_adam, af_heart, bm_george, etc.
        image_width: int = 1024,
        image_height: int = 576
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "scenes").mkdir(exist_ok=True)
        
        # Initialize generators
        self.audio_gen = AudioGenerator(voice=voice)
        self.image_gen = ImageGenerator(width=image_width, height=image_height)
        self.video_gen = VideoGenerator()
        self.combiner = SceneCombiner()
        self.stitcher = VideoStitcher()
    
    def generate_scene(self, scene: dict, scene_id: int) -> str:
        """
        Generate complete scene (audio + image + video).
        
        Args:
            scene: Scene dict with narration_text, visual_prompt, motion_prompt
            scene_id: Scene number
            
        Returns:
            Path to combined scene video
        """
        print(f"\n{'='*50}")
        print(f"🎬 SCENE {scene_id}")
        print(f"{'='*50}")
        
        # Paths
        audio_path = self.output_dir / "audio" / f"scene_{scene_id:03d}.wav"
        image_path = self.output_dir / "images" / f"scene_{scene_id:03d}.png"
        video_path = self.output_dir / "videos" / f"scene_{scene_id:03d}.mp4"
        scene_path = self.output_dir / "scenes" / f"scene_{scene_id:03d}.mp4"
        
        # Step 1: Generate Audio
        print(f"\n🎤 Generating audio...")
        print(f"   Text: {scene['narration_text'][:80]}...")
        self.audio_gen.generate(scene['narration_text'], str(audio_path))
        print(f"   ✓ Saved: {audio_path}")
        
        # Step 2: Generate Image
        print(f"\n🖼️  Generating image...")
        print(f"   Prompt: {scene['visual_prompt'][:80]}...")
        self.image_gen.generate(scene['visual_prompt'], str(image_path))
        print(f"   ✓ Saved: {image_path}")
        
        # Step 3: Generate Video from Image
        print(f"\n🎥 Generating video...")
        print(f"   Motion: {scene['motion_prompt'][:80]}...")
        self.video_gen.generate(str(image_path), scene['motion_prompt'], str(video_path))
        print(f"   ✓ Saved: {video_path}")
        
        # Step 4: Combine Audio + Video
        print(f"\n🔗 Combining audio + video...")
        self.combiner.combine(str(video_path), str(audio_path), str(scene_path))
        print(f"   ✓ Saved: {scene_path}")
        
        return str(scene_path)
    
    def generate_from_screenplay(self, screenplay_path: str, output_name: str = None) -> str:
        """
        Generate complete video from screenplay JSON.
        
        Args:
            screenplay_path: Path to screenplay JSON file
            output_name: Name for final video (default: based on input)
            
        Returns:
            Path to final video
        """
        # Load screenplay
        with open(screenplay_path, "r") as f:
            screenplay = json.load(f)
        
        scenes = screenplay.get("scenes", [])
        print(f"\n📽️  Starting video generation for {len(scenes)} scenes")
        print(f"   Output directory: {self.output_dir}")
        
        # Generate each scene
        scene_paths = []
        for scene in scenes:
            scene_id = scene.get("scene_id", len(scene_paths) + 1)
            scene_path = self.generate_scene(scene, scene_id)
            scene_paths.append(scene_path)
        
        # Unload models for stitching
        memory_manager.unload_current()
        
        # Stitch all scenes
        if output_name is None:
            output_name = Path(screenplay_path).stem.replace("_screenplay", "")
        
        final_path = self.output_dir / f"{output_name}_final.mp4"
        self.stitcher.stitch(scene_paths, str(final_path), transition="fade")
        
        print(f"\n{'='*50}")
        print(f"✅ VIDEO COMPLETE: {final_path}")
        print(f"{'='*50}")
        
        return str(final_path)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate video from screenplay")
    parser.add_argument("screenplay", help="Path to screenplay JSON")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--name", help="Name for final video")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument(
        "--voice", 
        default="am_adam",
        choices=["am_adam", "am_michael", "af_heart", "af_bella", "bm_george", "bf_emma"],
        help="Voice ID for TTS: am_adam (US male), af_heart (US female), bm_george (UK male)"
    )
    
    args = parser.parse_args()
    
    pipeline = VideoPipeline(
        output_dir=args.output_dir,
        voice=args.voice,
        image_width=args.width,
        image_height=args.height
    )
    
    pipeline.generate_from_screenplay(args.screenplay, args.name)


if __name__ == "__main__":
    main()
