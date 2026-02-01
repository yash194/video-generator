"""
Test script for Wan2.1 Text-to-Video generation.

Usage:
    python test_wan.py "Your video prompt here" -o output.mp4
    
Example:
    python test_wan.py "A golden DNA helix rotating slowly in a dark laboratory" -o test_dna.mp4
"""

import torch
import argparse
from diffusers import WanPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Test Wan2.1 T2V model")
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("-o", "--output", default="test_video.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames (default: 81 = ~5s)")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second (default: 16)")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps (default: 40)")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (default: 5.0)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 Wan2.1 Text-to-Video Test")
    print("=" * 60)
    print(f"📝 Prompt: {args.prompt}")
    print(f"🎞️  Frames: {args.frames} ({args.frames / args.fps:.1f}s at {args.fps}fps)")
    print(f"⚙️  Steps: {args.steps}")
    print(f"📊 Guidance: {args.guidance}")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading Wan2.1-T2V-14B...")
    print(f"💾 VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free / {torch.cuda.mem_get_info()[1]/1e9:.1f}GB total")
    
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
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
            negative_prompt="static, frozen, blurry, low quality, distorted, jittery, unnatural motion",
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=720,
            width=1280
        )
        frames = output.frames[0]
    
    # Export video
    print(f"\n💾 Saving to {args.output}...")
    export_to_video(frames, args.output, fps=args.fps)
    
    print(f"\n✅ Done! Video saved to: {args.output}")
    print(f"📊 Video: {args.frames} frames, {args.frames / args.fps:.1f}s, {args.fps}fps")


if __name__ == "__main__":
    main()
