import os
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

# Initialize Whisper model (lazy-loaded on first use)
_model = None

def get_whisper_model(model_size="base"):
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device="auto", compute_type="auto")
    return _model

def get_duration(video_path):
    """Get video duration in seconds"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(result.stdout.strip())
    except:
        return None

@app.route("/remove-silence", methods=["POST"])
def remove_silence():
    data = request.json
    video_path = data.get("video_path")
    keep_silence_up_to = data.get("keep_silence_up_to", 0.3)
    save_folder = data.get("save_folder", "./output")
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Invalid video_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    output_path = os.path.join(save_folder, f"{video_name}_trimmed.mp4")
    
    # Get input duration
    input_duration = get_duration(video_path)
    
    # Run auto-editor with video stream copy (lossless)
    cmd = [
        "auto-editor", video_path, "-o", output_path,
        "--margin", f"{keep_silence_up_to}sec", "--no-open"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        
        output_duration = get_duration(output_path)
        input_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        return jsonify({
            "status": "success",
            "video_path": str(video_path),
            "keep_silence_up_to": keep_silence_up_to,
            "save_folder": str(save_folder),
            "output_path": str(output_path),
            "input_duration_sec": round(input_duration, 2) if input_duration else None,
            "output_duration_sec": round(output_duration, 2) if output_duration else None,
            "time_removed_sec": round(input_duration - output_duration, 2) if input_duration and output_duration else None,
            "input_size_mb": round(input_size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "storage_saved_mb": round(input_size_mb - output_size_mb, 2),
        }), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "auto-editor failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    audio_path = data.get("audio_path")
    model_size = data.get("model_size", "base")
    save_folder = data.get("save_folder", "./output")
    language = data.get("language", None)
    
    if not audio_path or not Path(audio_path).exists():
        return jsonify({"error": "Invalid audio_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        model = get_whisper_model(model_size)
        segments, info = model.transcribe(audio_path, language=language, word_level=True)
        segments = list(segments)  # Convert generator to list for multiple iterations
        
        audio_name = Path(audio_path).stem
        vtt_path = os.path.join(save_folder, f"{audio_name}.vtt")
        
        # Generate VTT with word-level timestamps
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        start = _ms_to_vtt(word.start)
                        end = _ms_to_vtt(word.end)
                        f.write(f"{start} --> {end}\n{word.word.strip()}\n\n")
                else:
                    start = _ms_to_vtt(segment.start)
                    end = _ms_to_vtt(segment.end)
                    f.write(f"{start} --> {end}\n{segment.text.strip()}\n\n")
        
        vtt_size_kb = Path(vtt_path).stat().st_size / 1024
        audio_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        word_count = sum(len(s.words) if s.words else len(s.text.split()) for s in segments)
        
        return jsonify({
            "status": "success",
            "audio_path": str(audio_path),
            "model_size": model_size,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "vtt_path": str(vtt_path),
            "vtt_size_kb": round(vtt_size_kb, 2),
            "audio_size_mb": round(audio_size_mb, 2),
            "duration_sec": round(info.duration, 2),
            "word_count": word_count,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _ms_to_vtt(seconds):
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
