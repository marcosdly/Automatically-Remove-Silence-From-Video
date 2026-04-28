import os
import subprocess
import time
import psutil
import json
import re
from pathlib import Path
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from llama_cpp import Llama

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

@app.route("/optimize-audio", methods=["POST"])
def optimize_audio():
    data = request.json
    input_path = data.get("input_path")
    save_folder = data.get("save_folder", "./output")
    
    if not input_path or not Path(input_path).exists():
        return jsonify({"error": "Invalid input_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        input_name = Path(input_path).stem
        output_path = os.path.join(save_folder, f"{input_name}_optimized.m4a")
        
        # Get input duration and original size
        input_duration = get_duration(input_path)
        input_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
        
        # FFmpeg filters for transcription optimization:
        # - anlmdn: Noise reduction (Adaptive Noise Reduction)
        # - loudnorm: Normalize audio levels for consistency
        # - resample to 16kHz mono (optimal for Whisper)
        cmd = [
            "ffmpeg", "-i", input_path,
            "-af", "anlmdn=f=13:t=0.0001,loudnorm",
            "-ar", "16000", "-ac", "1",
            "-c:a", "aac", "-q:a", "8",
            "-y", output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        return jsonify({
            "status": "success",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_duration_sec": round(input_duration, 2) if input_duration else None,
            "input_size_mb": round(input_size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "compression_ratio": round(input_size_mb / output_size_mb, 2) if output_size_mb > 0 else 0,
            "storage_saved_mb": round(input_size_mb - output_size_mb, 2),
            "optimizations_applied": [
                "Noise reduction (adaptive)",
                "Level normalization",
                "16kHz mono (Whisper optimized)",
                "AAC encoding (efficient)"
            ]
        }), 200
    except subprocess.CalledProcessError:
        return jsonify({"error": "ffmpeg processing failed"}), 500
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

def _parse_vtt_time(time_str):
    """Parse VTT timestamp to seconds"""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

def _extract_score(evaluation_line):
    """Extract numeric virality score from evaluation line"""
    try:
        # Format: "[30s @ 0s-30s] | score_text"
        if "] | " in evaluation_line:
            score_part = evaluation_line.split("] | ")[1]
            # Extract first number (e.g., "7/10 - engaging" -> 7)
            import re
            match = re.search(r'^\d+', score_part)
            if match:
                return int(match.group())
    except:
        pass
    return 0

def _vtt_to_ass(vtt_path, save_folder):
    """Convert VTT with word-level timing to ASS format with styling"""
    ass_path = os.path.join(save_folder, f"{Path(vtt_path).stem}.ass")
    
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    subtitles = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if " --> " in line and not line.startswith("WEBVTT"):
            parts = line.split(" --> ")
            start = parts[0].strip()
            end = parts[1].strip()
            i += 1
            text = []
            while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                text.append(lines[i].strip())
                i += 1
            subtitles.append({"start": start, "end": end, "text": " ".join(text)})
        else:
            i += 1
    
    # Create ASS file with styling
    ass_header = """[Script Info]
Title: Shorts
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for sub in subtitles:
            text = sub["text"].replace('"', '\\"')
            f.write(f"Dialogue: 0,{sub['start']},{sub['end']},Default,,0,0,0,,{text}\n")
    
    return ass_path

@app.route("/filter-best-candidates", methods=["POST"])
def filter_best_candidates():
    data = request.json
    evaluations_path = data.get("evaluations_path")
    min_score = data.get("min_score", 6)
    top_n = data.get("top_n", 10)
    save_folder = data.get("save_folder", "./output")
    
    if not evaluations_path or not Path(evaluations_path).exists():
        return jsonify({"error": "Invalid evaluations_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse evaluations
        candidates = []
        with open(evaluations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "] | " in line:
                    score = _extract_score(line)
                    if score >= min_score:
                        candidates.append({"line": line, "score": score})
        
        if not candidates:
            return jsonify({"error": "No candidates meet min_score threshold"}), 400
        
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_n]
        
        # Save filtered results
        eval_name = Path(evaluations_path).stem.replace("_evaluations", "")
        output_path = os.path.join(save_folder, f"{eval_name}_top_candidates.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for c in top_candidates:
                f.write(c["line"] + "\n")
        
        output_size_kb = Path(output_path).stat().st_size / 1024
        
        return jsonify({
            "status": "success",
            "evaluations_path": str(evaluations_path),
            "min_score": min_score,
            "top_n": top_n,
            "total_evaluated": len(candidates) + sum(1 for line in open(evaluations_path) if line.strip() and "] | " in line and _extract_score(line) < min_score),
            "candidates_filtered": len(candidates),
            "candidates_selected": len(top_candidates),
            "top_scores": [c["score"] for c in top_candidates],
            "output_path": str(output_path),
            "output_size_kb": round(output_size_kb, 2),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/render-shorts", methods=["POST"])
def render_shorts():
    data = request.json
    video_path = data.get("video_path")
    vtt_path = data.get("vtt_path")
    title = data.get("title", "")
    save_folder = data.get("save_folder", "./output")
    output_width = data.get("output_width", 1080)
    output_height = data.get("output_height", 1920)
    blur_sigma = data.get("blur_sigma", 50)
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Invalid video_path"}), 400
    if not vtt_path or not Path(vtt_path).exists():
        return jsonify({"error": "Invalid vtt_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    try:
        # Get video dimensions
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height,duration", 
                     "-of", "csv=p=0", video_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
        vw, vh, duration = map(float, probe_result.split(','))
        vw, vh = int(vw), int(vh)
        
        # Calculate scale to fit video without distortion
        target_ratio = output_width / output_height
        video_ratio = vw / vh
        
        if video_ratio > target_ratio:  # Landscape wider than 4:3
            scale_w = int(output_height * video_ratio)
            scale_h = output_height
        else:
            scale_w = output_width
            scale_h = int(output_width / video_ratio)
        
        offset_x = (output_width - scale_w) // 2
        offset_y = (output_height - scale_h) // 2
        
        # Convert VTT to ASS with styled subtitles
        ass_path = _vtt_to_ass(vtt_path, save_folder)
        
        # FFmpeg complex filter: split video, blur one copy as background, overlay main on top, add subtitles
        combined_filter = (
            f"[0:v]split=2[bg][main];"
            f"[bg]scale={output_width}:{output_height},gblur=sigma={blur_sigma}[blurred];"
            f"[main]scale={scale_w}:{scale_h}[scaled];"
            f"[blurred][scaled]overlay={offset_x}:{offset_y}[final]"
        )
        
        output_path = os.path.join(save_folder, f"{Path(video_path).stem}_shorts.mp4")
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", combined_filter,
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "96k",
            "-y", output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        
        end_time = time.time()
        output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        return jsonify({
            "status": "success",
            "video_path": str(video_path),
            "vtt_path": str(vtt_path),
            "title": title,
            "output_width": output_width,
            "output_height": output_height,
            "blur_sigma": blur_sigma,
            "input_video_width": vw,
            "input_video_height": vh,
            "scaled_width": scale_w,
            "scaled_height": scale_h,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "output_path": str(output_path),
            "output_size_mb": round(output_size_mb, 2),
            "duration_sec": round(duration, 2),
            "processing_time_sec": round(end_time - start_time, 2),
        }), 200
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-metadata", methods=["POST"])
def generate_metadata():
    data = request.json
    window_text = data.get("window_text")
    model_path = data.get("model_path")
    save_folder = data.get("save_folder", "./output")
    
    if not window_text or not model_path or not Path(model_path).exists():
        return jsonify({"error": "Invalid window_text or model_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        model = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        
        # Generate metadata in one efficient prompt
        prompt = f"""Analyze this video content snippet and provide metadata in this exact format:
TAGS: tag1, tag2, tag3
KEYWORDS: keyword1, keyword2, keyword3, keyword4
TITLE: short title
DESCRIPTION: 1-2 sentence description
SHORT: brief 1-line description for display

Content: {window_text[:300]}

"""
        
        output = model(prompt, max_tokens=150, temperature=0.3, stop=["Content:"])
        response_text = output["choices"][0]["text"].strip()
        
        # Parse metadata from response
        metadata = {
            "tags": [],
            "keywords": [],
            "title": "",
            "description": "",
            "short_description": ""
        }
        
        for line in response_text.split("\n"):
            if line.startswith("TAGS:"):
                metadata["tags"] = [t.strip() for t in line.replace("TAGS:", "").split(",")]
            elif line.startswith("KEYWORDS:"):
                metadata["keywords"] = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]
            elif line.startswith("TITLE:"):
                metadata["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                metadata["description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("SHORT:"):
                metadata["short_description"] = line.replace("SHORT:", "").strip()
        
        del model
        
        # Save metadata
        import hashlib
        text_hash = hashlib.md5(window_text.encode()).hexdigest()[:8]
        output_path = os.path.join(save_folder, f"metadata_{text_hash}.json")
        
        import json
        metadata_with_context = {
            **metadata,
            "window_text": window_text,
            "model_path": str(model_path),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_with_context, f, indent=2, ensure_ascii=False)
        
        output_size_kb = Path(output_path).stat().st_size / 1024
        
        return jsonify({
            "status": "success",
            "window_text": window_text,
            "model_path": str(model_path),
            **metadata,
            "output_path": str(output_path),
            "output_size_kb": round(output_size_kb, 2),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate-virality", methods=["POST"])
def evaluate_virality():
    data = request.json
    windows_path = data.get("windows_path")
    model_path = data.get("model_path")
    save_folder = data.get("save_folder", "./output")
    
    if not windows_path or not Path(windows_path).exists():
        return jsonify({"error": "Invalid windows_path"}), 400
    if not model_path or not Path(model_path).exists():
        return jsonify({"error": "Invalid model_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    mem_start = psutil.Process().memory_info().rss / (1024 * 1024)
    
    try:
        # Load model
        print("Loading model...")
        model = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        
        # Parse windows file
        windows = []
        with open(windows_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract metadata and text: [30s @ 0s-30s] text...
                    if "] " in line:
                        meta, text = line.split("] ", 1)
                        windows.append({"meta": meta + "]", "text": text})
        
        if not windows:
            return jsonify({"error": "No windows found in file"}), 400
        
        # Evaluate each window
        evaluations = []
        prompt_template = "Rate the viral potential (1-10) of this video snippet and explain briefly in one sentence:\n\n{text}\n\nVirality Score:"
        
        for i, window in enumerate(windows):
            prompt = prompt_template.format(text=window["text"][:200])  # Truncate for speed
            output = model(prompt, max_tokens=50, temperature=0.3, stop=["Score:", "\n"])
            score_text = output["choices"][0]["text"].strip()
            evaluations.append(f"{window['meta']} | {score_text}")
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{len(windows)} windows")
        
        # Save evaluations
        windows_name = Path(windows_path).stem
        output_path = os.path.join(save_folder, f"{windows_name}_evaluations.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for eval_line in evaluations:
                f.write(eval_line + "\n")
        
        # Unload model
        del model
        
        # Calculate metrics
        end_time = time.time()
        mem_end = psutil.Process().memory_info().rss / (1024 * 1024)
        total_time = end_time - start_time
        
        output_size_kb = Path(output_path).stat().st_size / 1024
        windows_size_kb = Path(windows_path).stat().st_size / 1024
        
        return jsonify({
            "status": "success",
            "windows_path": str(windows_path),
            "model_path": str(model_path),
            "output_path": str(output_path),
            "windows_evaluated": len(evaluations),
            "output_size_kb": round(output_size_kb, 2),
            "windows_size_kb": round(windows_size_kb, 2),
            "total_time_sec": round(total_time, 2),
            "time_per_window_ms": round((total_time / len(evaluations) * 1000), 2) if evaluations else 0,
            "memory_peak_mb": round(max(mem_start, mem_end), 2),
            "memory_delta_mb": round(mem_end - mem_start, 2),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pipeline", methods=["POST"])
def pipeline():
    """Complete pipeline: remove silence → transcribe → evaluate → filter → render"""
    data = request.json
    video_path = data.get("video_path")
    model_path = data.get("model_path")
    save_folder = data.get("save_folder", "./output")
    keep_silence_up_to = data.get("keep_silence_up_to", 0.3)
    min_score = data.get("min_score", 6)
    top_n = data.get("top_n", 3)
    
    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "Invalid video_path"}), 400
    if not model_path or not Path(model_path).exists():
        return jsonify({"error": "Invalid model_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()
    results = {"steps": {}, "outputs": {}}
    
    try:
        video_name = Path(video_path).stem
        
        # Step 1: Remove silence
        t0 = time.time()
        trimmed = os.path.join(save_folder, f"{video_name}_trimmed.mp4")
        subprocess.run(["auto-editor", video_path, "-o", trimmed, "--margin", f"{keep_silence_up_to}sec", "--no-open"],
                      capture_output=True, check=True)
        results["steps"]["remove_silence"] = {"time": round(time.time() - t0, 2), "output": trimmed}
        
        # Step 2: Extract and optimize audio
        t0 = time.time()
        audio_raw = os.path.join(save_folder, f"{video_name}_audio.m4a")
        audio_opt = os.path.join(save_folder, f"{video_name}_audio_opt.m4a")
        subprocess.run(["ffmpeg", "-i", trimmed, "-q:a", "9", "-n", audio_raw], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-i", audio_raw, "-af", "anlmdn=f=13:t=0.0001,loudnorm", "-ar", "16000", "-ac", "1",
                       "-c:a", "aac", "-q:a", "8", "-y", audio_opt], capture_output=True, check=True)
        results["steps"]["audio"] = {"time": round(time.time() - t0, 2), "output": audio_opt}
        
        # Step 3: Transcribe
        t0 = time.time()
        model = get_whisper_model("base")
        segments, info = model.transcribe(audio_opt, word_level=True)
        segments = list(segments)
        vtt = os.path.join(save_folder, f"{video_name}.vtt")
        with open(vtt, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                if seg.words:
                    for word in seg.words:
                        f.write(f"{_ms_to_vtt(word.start)} --> {_ms_to_vtt(word.end)}\n{word.word.strip()}\n\n")
                else:
                    f.write(f"{_ms_to_vtt(seg.start)} --> {_ms_to_vtt(seg.end)}\n{seg.text.strip()}\n\n")
        results["steps"]["transcribe"] = {"time": round(time.time() - t0, 2), "language": info.language, "output": vtt}
        
        # Step 4: Extract windows
        t0 = time.time()
        with open(vtt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        subtitles = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if " --> " in line:
                parts = line.split(" --> ")
                start, end = _parse_vtt_time(parts[0]), _parse_vtt_time(parts[1])
                i += 1
                text = []
                while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                    text.append(lines[i].strip())
                    i += 1
                subtitles.append({"start": start, "end": end, "text": " ".join(text)})
            else:
                i += 1
        
        windows = []
        max_time = max(s["end"] for s in subtitles)
        for dur in range(30, 65, 5):
            t = 0
            while t + dur <= max_time:
                txt = " ".join([s["text"] for s in subtitles if s["start"] < t + dur and s["end"] > t])
                if txt.strip():
                    windows.append({"duration": dur, "start": t, "end": t + dur, "text": txt})
                t += 5
        
        win_file = os.path.join(save_folder, f"{video_name}_windows.txt")
        with open(win_file, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(f"[{w['duration']}s @ {w['start']:.0f}s-{w['end']:.0f}s] {w['text']}\n")
        results["steps"]["windows"] = {"time": round(time.time() - t0, 2), "count": len(windows), "output": win_file}
        
        # Step 5: Evaluate virality
        t0 = time.time()
        llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        evals = []
        for w in windows:
            p = f"Rate the viral potential (1-10) of this video snippet and explain briefly in one sentence:\n\n{w['text'][:200]}\n\nVirality Score:"
            out = llm(p, max_tokens=50, temperature=0.3, stop=["Score:", "\n"])
            score_text = out["choices"][0]["text"].strip()
            evals.append(f"[{w['duration']}s @ {w['start']:.0f}s-{w['end']:.0f}s] | {score_text}")
        del llm
        
        eval_file = os.path.join(save_folder, f"{video_name}_evals.txt")
        with open(eval_file, "w", encoding="utf-8") as f:
            for e in evals:
                f.write(e + "\n")
        results["steps"]["evaluate"] = {"time": round(time.time() - t0, 2), "count": len(evals), "output": eval_file}
        
        # Step 6: Filter best candidates
        t0 = time.time()
        candidates = [{"line": e, "score": _extract_score(e), "window": w} 
                     for e, w in zip(evals, windows) if _extract_score(e) >= min_score]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:top_n]
        
        cand_file = os.path.join(save_folder, f"{video_name}_candidates.txt")
        with open(cand_file, "w", encoding="utf-8") as f:
            for c in top:
                f.write(c["line"] + "\n")
        results["steps"]["filter"] = {"time": round(time.time() - t0, 2), "selected": len(top), "output": cand_file}
        
        # Step 7: Generate metadata and render shorts
        t0 = time.time()
        llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "stream=width,height", "-of", "csv=p=0", trimmed],
                              capture_output=True, text=True).stdout.strip()
        vw, vh = map(int, probe.split(','))
        
        ass_path = _vtt_to_ass(vtt, save_folder)
        shorts_info = []
        
        for idx, cand in enumerate(top):
            # Generate metadata
            p = f"Analyze this video content snippet and provide metadata in this exact format:\nTAGS: tag1, tag2, tag3\nKEYWORDS: keyword1, keyword2, keyword3, keyword4\nTITLE: short title\nDESCRIPTION: 1-2 sentence description\nSHORT: brief 1-line description for display\n\nContent: {cand['window']['text'][:300]}\n\n"
            out = llm(p, max_tokens=150, temperature=0.3, stop=["Content:"])
            resp = out["choices"][0]["text"].strip()
            
            meta = {"tags": [], "keywords": [], "title": "", "description": "", "short_description": ""}
            for line in resp.split("\n"):
                if line.startswith("TAGS:"):
                    meta["tags"] = [t.strip() for t in line.replace("TAGS:", "").split(",")]
                elif line.startswith("KEYWORDS:"):
                    meta["keywords"] = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]
                elif line.startswith("TITLE:"):
                    meta["title"] = line.replace("TITLE:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    meta["description"] = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("SHORT:"):
                    meta["short_description"] = line.replace("SHORT:", "").strip()
            
            # Render shorts
            out_w, out_h = 1080, 1920
            ratio = out_w / out_h
            vid_ratio = vw / vh
            scale_w = int(out_h * vid_ratio) if vid_ratio > ratio else out_w
            scale_h = int(out_w / vid_ratio) if vid_ratio > ratio else out_h
            off_x, off_y = (out_w - scale_w) // 2, (out_h - scale_h) // 2
            
            filt = f"[0:v]split=2[bg][main];[bg]scale={out_w}:{out_h},gblur=sigma=50[blurred];[main]scale={scale_w}:{scale_h}[scaled];[blurred][scaled]overlay={off_x}:{off_y}[final]"
            shorts = os.path.join(save_folder, f"{video_name}_shorts_{idx+1}.mp4")
            
            subprocess.run(["ffmpeg", "-i", trimmed, "-vf", filt, "-vf", f"ass={ass_path}",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "96k",
                           "-y", shorts], capture_output=True, check=True, timeout=600)
            
            shorts_info.append({
                "rank": idx + 1,
                "virality_score": cand["score"],
                "title": meta["title"],
                "short_description": meta["short_description"],
                "tags": meta["tags"],
                "output_path": str(shorts)
            })
        
        del llm
        results["outputs"]["shorts"] = shorts_info
        results["steps"]["render"] = {"time": round(time.time() - t0, 2), "count": len(shorts_info)}
        
        # Summary
        results["video_path"] = str(video_path)
        results["model_path"] = str(model_path)
        results["save_folder"] = str(save_folder)
        results["total_time_sec"] = round(time.time() - pipeline_start, 2)
        results["status"] = "success"
        
        return jsonify(results), 200
    
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract-subtitle-windows", methods=["POST"])
def extract_subtitle_windows():
    data = request.json
    vtt_path = data.get("vtt_path")
    save_folder = data.get("save_folder", "./output")
    
    if not vtt_path or not Path(vtt_path).exists():
        return jsonify({"error": "Invalid vtt_path"}), 400
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse VTT file
        with open(vtt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        subtitles = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if " --> " in line:
                parts = line.split(" --> ")
                start = _parse_vtt_time(parts[0])
                end = _parse_vtt_time(parts[1])
                i += 1
                text = []
                while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                    text.append(lines[i].strip())
                    i += 1
                subtitles.append({"start": start, "end": end, "text": " ".join(text)})
            else:
                i += 1
        
        if not subtitles:
            return jsonify({"error": "No subtitles found in VTT"}), 400
        
        # Extract windows: 30s-60s duration in 5s intervals
        windows = []
        duration_range = range(30, 65, 5)  # 30, 35, 40, 45, 50, 55, 60
        max_time = max(s["end"] for s in subtitles)
        
        for window_duration in duration_range:
            start_time = 0
            while start_time + window_duration <= max_time:
                window_end = start_time + window_duration
                window_text = " ".join([
                    s["text"] for s in subtitles
                    if s["start"] < window_end and s["end"] > start_time
                ])
                if window_text.strip():
                    windows.append({
                        "duration": window_duration,
                        "start": start_time,
                        "end": window_end,
                        "text": window_text
                    })
                start_time += 5  # 5s interval
        
        # Save windows as plain text
        vtt_name = Path(vtt_path).stem
        output_path = os.path.join(save_folder, f"{vtt_name}_windows.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(f"[{w['duration']}s @ {w['start']:.0f}s-{w['end']:.0f}s] {w['text']}\n")
        
        output_size_kb = Path(output_path).stat().st_size / 1024
        vtt_size_kb = Path(vtt_path).stat().st_size / 1024
        
        return jsonify({
            "status": "success",
            "vtt_path": str(vtt_path),
            "output_path": str(output_path),
            "window_count": len(windows),
            "duration_ranges": list(duration_range),
            "max_subtitle_time_sec": round(max_time, 2),
            "output_size_kb": round(output_size_kb, 2),
            "vtt_size_kb": round(vtt_size_kb, 2),
            "storage_ratio": round(output_size_kb / vtt_size_kb, 2) if vtt_size_kb > 0 else 0,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

