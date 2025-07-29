import asyncio
import base64
import json
import io
from typing import Optional

import numpy as np
import soundfile as sf
import websockets

# These modules are part of the Nvidia Audio2Face gRPC API.
# Install the Audio2Face SDK and ensure the generated Python
# stubs are available on the PYTHONPATH.
import grpc
import audio2face_pb2
import audio2face_pb2_grpc


def push_audio_track(
    url: str, audio_data: np.ndarray, samplerate: int, instance_name: str
) -> None:
    """Send PCM audio to Audio2Face for playback."""
    block_until_playback_is_finished = True
    with grpc.insecure_channel(url) as channel:
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        request = audio2face_pb2.PushAudioRequest()
        request.audio_data = audio_data.astype(np.float32).tobytes()
        request.samplerate = samplerate
        request.instance_name = instance_name
        request.block_until_playback_is_finished = block_until_playback_is_finished
        print("Sending audio to Audio2Face...")
        response = stub.PushAudio(request)
        if response.success:
            print("Audio playback succeeded")
        else:
            print(f"Playback error: {response.message}")


def decode_audio_payload(audio_b64: str) -> tuple[np.ndarray, int]:
    """Decode base64 WAV audio to float32 PCM."""
    audio_bytes = base64.b64decode(audio_b64)
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
    return data.astype(np.float32) / 32768.0, samplerate


async def handle_server_messages(ws, audio2face_url: str) -> None:
    async for message in ws:
        data = json.loads(message)
        msg_type = data.get("type")
        if msg_type == "audio" and data.get("audio"):
            try:
                pcm, sr = decode_audio_payload(data["audio"])
                push_audio_track(
                    url=audio2face_url,
                    audio_data=pcm,
                    samplerate=sr,
                    instance_name="/World/audio2face/PlayerStreaming",
                )
            except Exception as e:
                print(f"Failed to send audio: {e}")
        elif msg_type == "full-text":
            print(f"AI: {data.get('text')}")
        elif msg_type == "control" and data.get("text") == "start-mic":
            print("Connection ready. Type your message below.")
        else:
            # Unhandled messages are printed for debugging
            print(f"[DEBUG] {data}")


async def user_input_loop(ws) -> None:
    loop = asyncio.get_event_loop()
    while True:
        text: Optional[str] = await loop.run_in_executor(None, input, "You: ")
        if text is None:
            continue
        if text.lower() == "exit":
            await ws.close()
            break
        if text.lower() == "interrupt":
            await ws.send(json.dumps({"type": "interrupt-signal", "text": ""}))
        else:
            await ws.send(json.dumps({"type": "text-input", "text": text}))


async def main() -> None:
    ws_url = "ws://localhost:12393/client-ws"
    audio2face_url = "localhost:50051"
    async with websockets.connect(ws_url) as ws:
        await asyncio.gather(
            handle_server_messages(ws, audio2face_url),
            user_input_loop(ws),
        )


if __name__ == "__main__":
    asyncio.run(main())