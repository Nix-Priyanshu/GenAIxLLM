#!/usr/bin/env python3
"""Batch run: process every audio file in a folder through the agent, saving
text + audio replies and the document chunks used, into sample_outputs/.

Usage: python run.py --audio_dir path/to/audio --out_dir sample_outputs
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agent import ZentroAgent
from src.voice import voice_turn

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--out_dir", default="sample_outputs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.audio_dir)
                    if os.path.splitext(f)[1].lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files found in {args.audio_dir}")
        return

    agent = ZentroAgent()  # one agent per batch; use a fresh one per caller in production
    manifest = []
    for fname in files:
        in_path = os.path.join(args.audio_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_audio = os.path.join(args.out_dir, f"{stem}_reply.wav")

        heard, reply = voice_turn(in_path, agent, out_audio)
        sources = [c.section_id for c in agent.kb.retrieve(heard, k=5)]

        record = {"file": fname, "heard": heard, "reply": reply,
                   "sources": sources, "reply_audio": out_audio}
        manifest.append(record)
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        print(f"{fname}: {heard!r} -> {reply[:80]}...")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(manifest)} replies saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
