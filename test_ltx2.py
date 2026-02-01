"""
Test script for LTX-2 Text-to-Video generation.

LTX-2 is the fastest open-source video model - generates 5s video in ~4s on H100!

Usage:
    python test_ltx2.py "Your video prompt here" -o output.mp4
    
Example:
    python test_ltx2.py "A golden DNA helix rotating slowly in a dark laboratory" -o test_dna.mp4
"""

import torch
import argparse
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Test LTX-2 T2V model")
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("-o", "--output", default="test_video.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames (default: 121 = ~5s at 24fps)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps (default: 40)")
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance scale (default: 3.0)")
    parser.add_argument("--height", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=704, help="Video width (default: 704)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚡ LTX-2 Text-to-Video Test (FASTEST)")
    print("=" * 60)
    print(f"📝 Prompt: {args.prompt}")
    print(f"🎞️  Frames: {args.frames} ({args.frames / args.fps:.1f}s at {args.fps}fps)")
    print(f"📐 Resolution: {args.width}x{args.height}")
    print(f"⚙️  Steps: {args.steps}")
    print(f"📊 Guidance: {args.guidance}")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading LTX-Video...")
    print(f"💾 VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free / {torch.cuda.mem_get_info()[1]/1e9:.1f}GB total")
    
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
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
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width
        )
        frames = output.frames[0]
    
    # Export video
    print(f"\n💾 Saving to {args.output}...")
    export_to_video(frames, args.output, fps=args.fps)
    
    print(f"\n✅ Done! Video saved to: {args.output}")
    print(f"📊 Video: {args.frames} frames, {args.frames / args.fps:.1f}s, {args.fps}fps, {args.width}x{args.height}")


if __name__ == "__main__":
    main()
