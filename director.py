"""
Director Agent - Scene-by-Scene Video Screenplay Generator

Process:
1. Break text into scene-sized chunks
2. Send each chunk to LLM for direction
3. Get 3 elements per scene: narration, visual_prompt, motion_prompt

Dependencies:
    pip install openai google-generativeai
"""

import os
import json
import re
from pathlib import Path

# LLM providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


SCENE_PROMPT = """You are a cinematic video director. Convert this content chunk into ONE video scene.

Content:
---
{chunk}
---

Return ONLY a JSON object with exactly 3 fields:

{{
    "narration_text": "The voiceover script for this scene (50-80 words, natural speaking style)",
    
    "visual_prompt": "Detailed image generation prompt: describe the main visual, environment, lighting, style. Include: photorealistic, 8k, cinematic. Be specific about what to show.",
    
    "motion_prompt": "Describe BOTH subject motion AND camera motion for video generation. Format: [SUBJECT MOTION] + [CAMERA MOTION] + [ENVIRONMENT]. Examples:
    - 'Light rays animate traveling through the lens, refracting and splitting into spectrum colors. Camera slowly pushes forward. Subtle dust particles float in the air.'
    - 'The mirror surface ripples slightly as the light beam bounces off. Camera orbits around the reflection point. Background stays static.'
    - 'Water droplets fall in slow motion creating expanding ripples. Camera tracks the falling droplet from above. Gentle mist rises from the surface.'
    - 'Diagram elements draw themselves on screen sequentially, labels fade in. Camera holds static with slight breathing motion.'
    - 'The pendulum swings smoothly left to right. Camera follows the motion with subtle parallax. Shadows move across the floor.'
    Be specific about what moves, how it moves, and the camera behavior. This will be used by AI video generators like Runway/Pika."
}}

Output ONLY valid JSON, no other text."""


class Director:
    """Scene-by-scene video director."""
    
    def __init__(self, provider: str = "auto", api_key: str = None):
        self.provider = None
        
        # Auto-detect
        if provider == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            elif os.environ.get("GOOGLE_API_KEY"):
                provider = "gemini"
        
        if provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key and OPENAI_AVAILABLE:
                self.client = openai.OpenAI(api_key=key)
                self.provider = "openai"
                print("✓ Director ready (OpenAI)")
        
        elif provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if key and GEMINI_AVAILABLE:
                genai.configure(api_key=key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.provider = "gemini"
                print("✓ Director ready (Gemini)")
        
        if not self.provider:
            raise ValueError("No LLM available. Set OPENAI_API_KEY or GOOGLE_API_KEY")
    
    def break_into_scenes(self, text: str, words_per_scene: int = 150) -> list:
        """
        Break text into scene-sized chunks.
        
        Uses paragraph boundaries when possible.
        Target: ~150 words per scene (roughly 20-30 seconds of narration)
        """
        # Split by paragraphs (double newlines or page markers)
        paragraphs = re.split(r'\n\s*\n|## Page \d+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        scenes = []
        current_chunk = []
        current_words = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # Skip very short paragraphs (headers, etc) by merging
            if para_words < 20 and current_chunk:
                current_chunk.append(para)
                current_words += para_words
                continue
            
            # If adding this paragraph exceeds limit, start new scene
            if current_words + para_words > words_per_scene and current_chunk:
                scenes.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words
        
        # Don't forget last chunk
        if current_chunk:
            scenes.append("\n\n".join(current_chunk))
        
        return scenes
    
    def direct_scene(self, chunk: str) -> dict:
        """Send one chunk to LLM and get scene direction."""
        prompt = SCENE_PROMPT.format(chunk=chunk)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content
                
            elif self.provider == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                )
                result = response.text
            
            return json.loads(result)
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            return {
                "narration_text": chunk[:200] + "...",
                "visual_prompt": "Educational scene, photorealistic, 8k",
                "motion_prompt": "slow zoom in, 5s"
            }
    
    def create_screenplay(self, text: str, words_per_scene: int = 150) -> list:
        """
        Process text and create full screenplay scene by scene.
        
        Args:
            text: Full extracted text
            words_per_scene: Approximate words per scene chunk
            
        Returns:
            List of scene dictionaries
        """
        # Step 1: Break into chunks
        chunks = self.break_into_scenes(text, words_per_scene)
        print(f"📝 Split into {len(chunks)} scenes")
        
        # Step 2: Process each chunk
        scenes = []
        for i, chunk in enumerate(chunks):
            print(f"🎬 Directing scene {i+1}/{len(chunks)}...", end=" ")
            
            scene = self.direct_scene(chunk)
            scene["scene_id"] = i + 1
            scene["source_text"] = chunk  # Keep original for reference
            
            scenes.append(scene)
            print("✓")
        
        return scenes
    
    def direct_from_file(self, input_path: str, output_path: str = None) -> str:
        """
        Read text file and create screenplay.
        
        Args:
            input_path: Path to extracted text file
            output_path: Output JSON path
            
        Returns:
            Path to output file
        """
        input_path = Path(input_path)
        
        # Read content
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create screenplay
        scenes = self.create_screenplay(text)
        
        # Prepare output
        screenplay = {
            "source_file": str(input_path),
            "total_scenes": len(scenes),
            "scenes": scenes
        }
        
        # Save
        if output_path is None:
            output_path = input_path.with_name(f"{input_path.stem}_screenplay.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(screenplay, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved: {output_path}")
        return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene-by-scene video director")
    parser.add_argument("input", help="Extracted text file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--words-per-scene", type=int, default=150, 
                        help="Target words per scene (default: 150)")
    parser.add_argument("--provider", choices=["openai", "gemini", "auto"], default="auto")
    
    args = parser.parse_args()
    
    director = Director(provider=args.provider)
    director.direct_from_file(args.input, args.output)
    
    print("\n🎬 Done!")


if __name__ == "__main__":
    main()
