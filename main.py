import argparse
from detector import train_model, predict_scam
from utils import audio_to_text

def main():
    parser = argparse.ArgumentParser(description="IVR Scam Call Detector")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--detect', type=str, help='Detect from text input')
    parser.add_argument('--audio', type=str, help='Detect from audio file')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.detect:
        result = predict_scam(args.detect)
        print(f"🔍 Result: {result.upper()}")
    elif args.audio:
        print("🎧 Converting audio to text...")
        text = audio_to_text(args.audio)
        if not text:
            print("❌ Could not recognize speech.")
        else:
            print(f"🗣 Transcribed Text: {text}")
            result = predict_scam(text)
            print(f"🔍 Result: {result.upper()}")
    else:
        print("❗ Use --train, --detect 'text', or --audio path_to_file")

if __name__ == "__main__":
    main()
