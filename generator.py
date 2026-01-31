"""
Video Generation Pipeline

Complete scene-by-scene video generation from screenplay JSON.
Uses open-source models with GPU memory management for 40GB VRAM.

Models:
- Audio: Parler-TTS (emotional speech)
- Image: SDXL 1.0 (high quality images)
- Video: CogVideoX-5B (image-to-video with motion)

Dependencies:
    pip install torch diffusers transformers accelerate
    pip install parler-tts soundfile
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
# AUDIO GENERATOR (Parler-TTS)
# ============================================================================

class AudioGenerator:
    """Generate emotional speech from text using Parler-TTS."""
    
    def __init__(
        self, 
        model_id: str = "parler-tts/parler-tts-large-v1",
        voice_description: str = "A male speaker with a clear, professional voice delivers the narration with moderate pace and natural intonation."
    ):
        self.model_id = model_id
        self.voice_description = voice_description
        self.model = None
        self.tokenizer = None
    
    def _load(self):
        """Load Parler-TTS model."""
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        return {"model": model, "tokenizer": tokenizer}
    
    def load(self):
        """Load model via memory manager."""
        result = memory_manager.load_model("Parler-TTS", self._load)
        self.model = result["model"]
        self.tokenizer = result["tokenizer"]
    
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
        
        if self.model is None:
            self.load()
        
        # Prepare inputs
        input_ids = self.tokenizer(
            self.voice_description, 
            return_tensors="pt"
        ).input_ids.to("cuda")
        
        prompt_input_ids = self.tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids.to("cuda")
        
        # Generate
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=1.0
            )
        
        # Save audio
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_path, audio_arr, self.model.config.sampling_rate)
        
        return output_path


# ============================================================================
# IMAGE GENERATOR (SDXL)
# ============================================================================

class ImageGenerator:
    """Generate images from visual prompts using SDXL."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        width: int = 1280,
        height: int = 720
    ):
        self.model_id = model_id
        self.width = width
        self.height = height
        self.pipe = None
    
    def _load(self):
        """Load SDXL pipeline."""
        from diffusers import StableDiffusionXLPipeline
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        return pipe
    
    def load(self):
        """Load model via memory manager."""
        self.pipe = memory_manager.load_model("SDXL", self._load)
    
    def generate(
        self, 
        prompt: str, 
        output_path: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy"
    ) -> str:
        """
        Generate image from visual prompt.
        
        Args:
            prompt: Visual description prompt
            output_path: Path to save image (.png)
            negative_prompt: Things to avoid
            
        Returns:
            Path to generated image
        """
        if self.pipe is None:
            self.load()
        
        # Generate image
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # Save
        image.save(output_path)
        return output_path


# ============================================================================
# VIDEO GENERATOR (CogVideoX)
# ============================================================================

class VideoGenerator:
    """Generate video from image + motion prompt using CogVideoX."""
    
    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-5b-I2V",
        num_frames: int = 49,  # ~6 seconds at 8fps
        fps: int = 8
    ):
        self.model_id = model_id
        self.num_frames = num_frames
        self.fps = fps
        self.pipe = None
    
    def _load(self):
        """Load CogVideoX pipeline."""
        from diffusers import CogVideoXImageToVideoPipeline
        
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        
        return pipe
    
    def load(self):
        """Load model via memory manager."""
        self.pipe = memory_manager.load_model("CogVideoX", self._load)
    
    def generate(
        self,
        image_path: str,
        motion_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate video from image with motion.
        
        Args:
            image_path: Path to input image
            motion_prompt: Description of motion/camera movement
            output_path: Path to save video (.mp4)
            
        Returns:
            Path to generated video
        """
        from PIL import Image
        from diffusers.utils import export_to_video
        
        if self.pipe is None:
            self.load()
        
        # Load input image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((720, 480))  # CogVideoX input size
        
        # Generate video frames
        with torch.no_grad():
            video_frames = self.pipe(
                prompt=motion_prompt,
                image=image,
                num_frames=self.num_frames,
                num_inference_steps=50,
                guidance_scale=6.0,
                use_dynamic_cfg=True
            ).frames[0]
        
        # Export to video
        export_to_video(video_frames, output_path, fps=self.fps)
        
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
        from moviepy.editor import VideoFileClip, AudioFileClip, vfx
        
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
                video = video.fx(vfx.loop, n=loops_needed)
            video = video.subclip(0, audio_duration)
        else:
            # Match audio to video duration
            if audio_duration > video_duration:
                audio = audio.subclip(0, video_duration)
            # If audio is shorter, video will have silence at end
        
        # Combine
        final = video.set_audio(audio)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium",
            verbose=False,
            logger=None
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
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        print(f"🎬 Stitching {len(scene_paths)} scenes...")
        
        clips = []
        for path in scene_paths:
            clip = VideoFileClip(path)
            
            if transition == "fade":
                clip = clip.fadein(0.5).fadeout(0.5)
            
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
            preset="medium",
            verbose=False,
            logger=None
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
        voice_description: str = None,
        image_width: int = 1280,
        image_height: int = 720
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "scenes").mkdir(exist_ok=True)
        
        # Initialize generators
        self.audio_gen = AudioGenerator(
            voice_description=voice_description or 
            "A professional male narrator with clear enunciation and engaging tone."
        )
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
        default="A professional male narrator with clear enunciation and engaging tone.",
        help="Voice description for TTS"
    )
    
    args = parser.parse_args()
    
    pipeline = VideoPipeline(
        output_dir=args.output_dir,
        voice_description=args.voice,
        image_width=args.width,
        image_height=args.height
    )
    
    pipeline.generate_from_screenplay(args.screenplay, args.name)


if __name__ == "__main__":
    main()
