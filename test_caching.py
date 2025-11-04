#!/usr/bin/env python3
"""
Simple test script to debug Chatterbox caching behavior.
This makes it easier to test without running the full orchestrator.
"""

import sys
import time
from tts_chatterbox_service import ChatterboxTTS

def main():
    print("=" * 60)
    print("CHATTERBOX CACHING DEBUG TEST")
    print("=" * 60)
    
    # Initialize with reference audio
    speaker_wav = "amitabh_cut.wav"
    print(f"\n[TEST] Initializing ChatterboxTTS with speaker_wav={speaker_wav}")
    print("-" * 60)
    
    tts = ChatterboxTTS(
        multilingual=False,
        speaker_wav=speaker_wav,
        device="mps",
        use_gpu=True,
        exaggeration=0.5,
        cfg=0.5,
    )
    
    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE - NOW TESTING SYNTHESIS")
    print("=" * 60)
    
    # Test 1: First synthesis
    print("\n[TEST] === REQUEST 1 ===")
    print("-" * 60)
    text1 = "Hello, this is the first test."
    start1 = time.time()
    audio1 = tts.synth(text1)
    elapsed1 = time.time() - start1
    print(f"[TEST] Request 1 completed in {elapsed1:.2f}s")
    
    # Test 2: Second synthesis (should be faster if caching works)
    print("\n[TEST] === REQUEST 2 ===")
    print("-" * 60)
    text2 = "Hello, this is the second test."
    start2 = time.time()
    audio2 = tts.synth(text2)
    elapsed2 = time.time() - start2
    print(f"[TEST] Request 2 completed in {elapsed2:.2f}s")
    
    # Test 3: Third synthesis
    print("\n[TEST] === REQUEST 3 ===")
    print("-" * 60)
    text3 = "Hello, this is the third test."
    start3 = time.time()
    audio3 = tts.synth(text3)
    elapsed3 = time.time() - start3
    print(f"[TEST] Request 3 completed in {elapsed3:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Request 1: {elapsed1:.2f}s")
    print(f"Request 2: {elapsed2:.2f}s (speedup: {elapsed1/elapsed2:.2f}x)")
    print(f"Request 3: {elapsed3:.2f}s (speedup: {elapsed1/elapsed3:.2f}x)")
    
    if elapsed2 < elapsed1 * 0.8:
        print("\n✅ SUCCESS: Caching is working! Requests 2+ are faster.")
    else:
        print("\n❌ PROBLEM: No speedup detected. Caching may not be working.")
        print("\nLook at the debug logs above to see which code branch was taken.")
        print("If you see 'Branch 2: using audio_prompt_path' on requests 2+, that's the problem.")

if __name__ == "__main__":
    main()