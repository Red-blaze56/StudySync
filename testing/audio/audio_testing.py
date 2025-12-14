from pathlib import Path
from typing import List
from langchain_core.documents import Document
from faster_whisper import WhisperModel

from src.Processors.audio_processor import load_audio

def test_audio():
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    path = Path(r"testing/audio/cc_vid.mp4")

    docs = load_audio(path,model)
    
    print("=" * 60)
    print("FILE:", path.name)
    print("Documents:", len(docs))

    if not docs:
        print("No docs return")
        return
    
    total_chars = sum(len(d.page_content) for d in docs)
    print("Total characters:", total_chars)
    print("Metadata sample:", docs[0].metadata)
    print("Preview:")
    print(docs[0].page_content[:300])
    print("=" * 60)

def main():
    print("Runnig Audio Pipeline Test....")
    test_audio()


if __name__ == "__main__":
    main()



