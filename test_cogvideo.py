"""
Test script for CogVideoX-5B Text-to-Video generation.

CogVideoX-5B features:
- 5B parameters, high quality
- Good motion consistency
- 6 seconds at 8fps (49 frames)
- 720x480 resolution
- ~16GB VRAM

Usage:
    python test_cogvideo.py "Your video prompt here" -o output.mp4
    
Example:
    python test_cogvideo.py "A golden DNA helix rotating slowly in a dark laboratory" -o test_dna.mp4
"""

import torch
import argparse
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Test CogVideoX-5B T2V model")
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("-o", "--output", default="test_video.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=49, help="Number of frames (default: 49 = ~6s at 8fps)")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second (default: 8)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale (default: 6.0)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 CogVideoX-5B Text-to-Video Test")
    print("=" * 60)
    print(f"📝 Prompt: {args.prompt}")
    print(f"🎞️  Frames: {args.frames} ({args.frames / args.fps:.1f}s at {args.fps}fps)")
    print(f"📐 Resolution: 720x480")
    print(f"⚙️  Steps: {args.steps}")
    print(f"📊 Guidance: {args.guidance}")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading CogVideoX-5B...")
    print(f"💾 VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free / {torch.cuda.mem_get_info()[1]/1e9:.1f}GB total")
    
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    # Enable memory optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print(f"✓ Model loaded")
    print(f"💾 VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free / {torch.cuda.mem_get_info()[1]/1e9:.1f}GB total")
    
    # Generate video
    print(f"\n🎥 Generating video...")
    
    with torch.no_grad():
        output = pipe(
            prompt=args.prompt,
            negative_prompt="blurry, low quality, distorted, jittery, unnatural motion",
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            use_dynamic_cfg=True
        )
        frames = output.frames[0]
    
    # Export video
    print(f"\n💾 Saving to {args.output}...")
    export_to_video(frames, args.output, fps=args.fps)
    
    print(f"\n✅ Done! Video saved to: {args.output}")
    print(f"📊 Video: {args.frames} frames, {args.frames / args.fps:.1f}s, {args.fps}fps, 720x480")


if __name__ == "__main__":
    main()
