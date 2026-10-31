"""Step 4: wrap the Step 3 agent with Gnani STT/TTS.

Setup (see capstone brief, 'Give it a voice'):
    1. Create a Gnani account + API key at app.gnani.ai/voice
    2. export GNANI_API_KEY=...
    3. pip install gnani-vachana
"""
import os
from src.agent import ZentroAgent

GNANI_LANGUAGE = os.environ.get("GNANI_LANGUAGE", "en-IN")
GNANI_VOICE = os.environ.get("GNANI_VOICE", "Kaveri")
GNANI_TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")  # must be named explicitly


def voice_turn(audio_path: str, agent: ZentroAgent, output_path: str):
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()

    heard = stt.transcribe(audio_path, language_code=GNANI_LANGUAGE)["transcript"]
    if not heard.strip():
        reply = "Sorry, I didn't catch that. Could you say it again?"
    else:
        reply = agent.respond(heard)

    tts.synthesize(reply, voice=GNANI_VOICE, model=GNANI_TTS_MODEL,
                    language=GNANI_LANGUAGE, output_file=output_path)
    return heard, reply
