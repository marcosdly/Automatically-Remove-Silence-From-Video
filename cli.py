#!/usr/bin/env python3
"""CLI script for testing the video shorts pipeline"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
from llama_cpp import Llama


def get_duration(video_path):
    """Get video duration in seconds"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return None


def ms_to_vtt(seconds):
    """Convert seconds to VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_vtt_time(time_str):
    """Parse VTT timestamp to seconds"""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def extract_score(evaluation_line):
    """Extract numeric virality score from evaluation line"""
    try:
        if "] | " in evaluation_line:
            score_part = evaluation_line.split("] | ")[1]
            match = re.search(r'^\d+', score_part)
            if match:
                return int(match.group())
    except:
        pass
    return 0


def vtt_to_ass(vtt_path, save_folder):
    """Convert VTT to ASS format with styling"""
    ass_path = Path(save_folder) / f"{Path(vtt_path).stem}.ass"
    
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    subtitles = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if " --> " in line and not line.startswith("WEBVTT"):
            parts = line.split(" --> ")
            start, end = parts[0].strip(), parts[1].strip()
            i += 1
            text = []
            while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                text.append(lines[i].strip())
                i += 1
            subtitles.append({"start": start, "end": end, "text": " ".join(text)})
        else:
            i += 1
    
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


def run_pipeline(video_path, model_path, save_folder="./output", keep_silence_up_to=0.3, 
                 min_score=6, whisper_model="base"):
    """Run complete pipeline with minimal logging"""
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()
    results = {"steps": {}, "outputs": {}, "params": {}}
    
    # Store parameters
    results["params"] = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "save_folder": str(save_folder),
        "keep_silence_up_to": keep_silence_up_to,
        "min_score": min_score,
        "whisper_model": whisper_model,
    }
    
    video_name = Path(video_path).stem
    save_folder = Path(save_folder)
    
    try:
        # Step 1: Remove silence
        print("Step 1/7: Removing silence...", end=" ", flush=True)
        t0 = time.time()
        trimmed = save_folder / f"{video_name}_trimmed.mp4"
        subprocess.run(
            ["auto-editor", str(video_path), "-o", str(trimmed), "--margin", f"{keep_silence_up_to}sec", 
             "--no-open", "--video-codec", "h264", "--audio-codec", "aac"],
            capture_output=True, check=True, timeout=600
        )
        # Verify output has video stream; if not, re-encode using ffmpeg
        probe_result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(trimmed)],
            capture_output=True, text=True, timeout=10
        )
        if not probe_result.stdout.strip() or "video" not in probe_result.stdout:
            print("(re-encoding to ensure video stream...)", end=" ", flush=True)
            trimmed_backup = trimmed.with_stem(f"{trimmed.stem}_backup")
            trimmed.rename(trimmed_backup)
            subprocess.run(
                ["ffmpeg", "-i", str(trimmed_backup), "-c:v", "libx264", "-preset", "ultrafast",
                 "-c:a", "aac", "-y", str(trimmed)],
                capture_output=True, check=True, timeout=600
            )
            trimmed_backup.unlink()
        step_time = time.time() - t0
        results["steps"]["remove_silence"] = {"time": round(step_time, 2), "output": str(trimmed)}
        print(f"✓ ({step_time:.1f}s)")
        
        # Step 2: Extract and optimize audio
        print("Step 2/7: Processing audio...", end=" ", flush=True)
        t0 = time.time()
        audio_raw = save_folder / f"{video_name}_audio.m4a"
        audio_opt = save_folder / f"{video_name}_audio_opt.wav"
        
        # Extract audio with proper stream selection
        subprocess.run(
            ["ffmpeg", "-i", str(trimmed), "-vn", "-c:a", "alac", "-y", str(audio_raw)],
            capture_output=True, check=True, timeout=300
        )
        
        # Optimize audio with loudness normalization
        subprocess.run(
            ["ffmpeg", "-i", str(trimmed),
             "-af", "highpass=f=80,lowpass=f=7600,afftdn=nf=-25,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,volume=3dB,loudnorm=I=-16:TP=-1.5:LRA=7",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(audio_opt)],
            capture_output=True, check=True, timeout=300
        )
        step_time = time.time() - t0
        results["steps"]["audio"] = {"time": round(step_time, 2), "output": str(audio_opt)}
        print(f"✓ ({step_time:.1f}s)")
        
        # Step 3: Transcribe
        print("Step 3/7: Transcribing...", end=" ", flush=True)
        t0 = time.time()
        model = WhisperModel(whisper_model, device="auto", compute_type="auto")
        segments, info = model.transcribe(str(audio_opt), word_timestamps=True)
        segments = list(segments)
        vtt = save_folder / f"{video_name}.vtt"
        with open(vtt, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                if seg.words:
                    for word in seg.words:
                        f.write(f"{ms_to_vtt(word.start)} --> {ms_to_vtt(word.end)}\n{word.word.strip()}\n\n")
                else:
                    f.write(f"{ms_to_vtt(seg.start)} --> {ms_to_vtt(seg.end)}\n{seg.text.strip()}\n\n")
        step_time = time.time() - t0
        results["steps"]["transcribe"] = {
            "time": round(step_time, 2),
            "language": info.language,
            "language_prob": round(info.language_probability, 3),
            "output": str(vtt)
        }
        print(f"✓ ({step_time:.1f}s)")
        del model
        
        # Step 4: Extract subtitle windows
        print("Step 4/7: Extracting windows...", end=" ", flush=True)
        t0 = time.time()
        with open(vtt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        subtitles = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if " --> " in line:
                parts = line.split(" --> ")
                start, end = parse_vtt_time(parts[0]), parse_vtt_time(parts[1])
                i += 1
                text = []
                while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                    text.append(lines[i].strip())
                    i += 1
                subtitles.append({"start": start, "end": end, "text": " ".join(text)})
            else:
                i += 1
        
        windows = []
        max_time = max(s["end"] for s in subtitles) if subtitles else 0
        for dur in range(30, 65, 5):
            t = 0
            while t + dur <= max_time:
                txt = " ".join([s["text"] for s in subtitles if s["start"] < t + dur and s["end"] > t])
                if txt.strip():
                    windows.append({"duration": dur, "start": t, "end": t + dur, "text": txt})
                t += 5
        
        win_file = save_folder / f"{video_name}_windows.txt"
        with open(win_file, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(f"[{w['duration']}s @ {w['start']:.0f}s-{w['end']:.0f}s] {w['text']}\n")
        step_time = time.time() - t0
        results["steps"]["windows"] = {"time": round(step_time, 2), "count": len(windows), "output": str(win_file)}
        print(f"✓ ({step_time:.1f}s)")
        
        # Step 5: Evaluate virality
        print("Step 5/7: Evaluating virality...", end=" ", flush=True)
        t0 = time.time()
        llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        evals = []
        for i, w in enumerate(windows):
            p = f"Rate the viral potential (1-10) of this video snippet and explain briefly in one sentence:\n\n{w['text'][:200]}\n\nVirality Score:"
            out = llm(p, max_tokens=50, temperature=0.3, stop=["Score:", "\n"])
            score_text = out["choices"][0]["text"].strip()
            evals.append(f"[{w['duration']}s @ {w['start']:.0f}s-{w['end']:.0f}s] | {score_text}")
        del llm
        
        eval_file = save_folder / f"{video_name}_evals.txt"
        with open(eval_file, "w", encoding="utf-8") as f:
            for e in evals:
                f.write(e + "\n")
        step_time = time.time() - t0
        results["steps"]["evaluate"] = {"time": round(step_time, 2), "count": len(evals), "output": str(eval_file)}
        print(f"✓ ({step_time:.1f}s)")
        
        # Step 6: Filter best candidates
        print("Step 6/7: Filtering candidates...", end=" ", flush=True)
        t0 = time.time()
        candidates = [{"line": e, "score": extract_score(e), "window": w}
                     for e, w in zip(evals, windows) if extract_score(e) >= min_score]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates
        
        cand_file = save_folder / f"{video_name}_candidates.txt"
        with open(cand_file, "w", encoding="utf-8") as f:
            for c in top:
                f.write(c["line"] + "\n")
        step_time = time.time() - t0
        results["steps"]["filter"] = {
            "time": round(step_time, 2),
            "total_evaluated": len(candidates) + sum(1 for e in evals if extract_score(e) < min_score),
            "selected": len(top),
            "top_scores": [c["score"] for c in top],
            "output": str(cand_file)
        }
        print(f"✓ ({step_time:.1f}s)")
        
        # Step 7: Render shorts
        print("Step 7/7: Rendering shorts...", end=" ", flush=True)
        t0 = time.time()
        llm = Llama(model_path=model_path, n_gpu_layers=35, n_ctx=512, verbose=False)
        
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(trimmed)],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
        vw, vh = map(int, probe.split(','))
        
        ass_path = vtt_to_ass(vtt, save_folder)
        shorts_info = []
        
        for idx, cand in enumerate(top):
            # Generate metadata
            p = f"Analyze this video content snippet and provide metadata in this exact format:\nTAGS: tag1, tag2, tag3\nKEYWORDS: keyword1, keyword2, keyword3, keyword4\nTITLE: short title\nDESCRIPTION: 1-2 sentence description\n\nContent: {cand['window']['text'][:300]}\n\n"
            out = llm(p, max_tokens=100, temperature=0.3, stop=["Content:"])
            resp = out["choices"][0]["text"].strip()
            
            meta = {"tags": [], "keywords": [], "title": "", "description": ""}
            for line in resp.split("\n"):
                if line.startswith("TAGS:"):
                    meta["tags"] = [t.strip() for t in line.replace("TAGS:", "").split(",")]
                elif line.startswith("KEYWORDS:"):
                    meta["keywords"] = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]
                elif line.startswith("TITLE:"):
                    meta["title"] = line.replace("TITLE:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    meta["description"] = line.replace("DESCRIPTION:", "").strip()
            
            # Render shorts
            out_w, out_h = 1080, 1920
            ratio = out_w / out_h
            vid_ratio = vw / vh
            scale_w = int(out_h * vid_ratio) if vid_ratio > ratio else out_w
            scale_h = int(out_w / vid_ratio) if vid_ratio > ratio else out_h
            off_x = (out_w - scale_w) // 2
            off_y = (out_h - scale_h) // 2
            
            filt = f"[0:v]split=2[bg][main];[bg]scale={out_w}:{out_h},gblur=sigma=50[blurred];[main]scale={scale_w}:{scale_h}[scaled];[blurred][scaled]overlay={off_x}:{off_y}[final]"
            shorts = save_folder / f"{video_name}_shorts_{idx+1}.mp4"
            
            subprocess.run(
                ["ffmpeg", "-i", str(trimmed), "-vf", filt, "-vf", f"ass={str(ass_path)}",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "96k", "-y", str(shorts)],
                capture_output=True, check=True, timeout=600
            )
            
            output_size_mb = shorts.stat().st_size / (1024 * 1024)
            shorts_info.append({
                "rank": idx + 1,
                "score": cand["score"],
                "title": meta["title"],
                "tags": meta["tags"],
                "output_path": str(shorts),
                "size_mb": round(output_size_mb, 2)
            })
        
        del llm
        results["outputs"]["shorts"] = shorts_info
        step_time = time.time() - t0
        results["steps"]["render"] = {"time": round(step_time, 2), "count": len(shorts_info)}
        print(f"✓ ({step_time:.1f}s)")
        
        # Summary
        results["total_time_sec"] = round(time.time() - pipeline_start, 2)
        results["status"] = "success"
        
        return results
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results


def main():
    parser = argparse.ArgumentParser(description="Video shorts pipeline CLI")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("model_path", help="Path to LLM model")
    parser.add_argument("-o", "--output", default="./output", help="Output folder (default: ./output)")
    parser.add_argument("-s", "--silence", type=float, default=0.3, help="Keep silence up to (default: 0.3)")
    parser.add_argument("-m", "--min-score", type=int, default=6, help="Minimum virality score (default: 6)")
    parser.add_argument("-w", "--whisper", default="base", help="Whisper model size (default: base)")
    parser.add_argument("-j", "--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(
            args.video_path,
            args.model_path,
            args.output,
            args.silence,
            args.min_score,
            args.whisper
        )
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            # Pretty print results
            print("\n" + "="*60)
            print("PIPELINE RESULTS")
            print("="*60)
            
            if results["status"] == "success":
                print("\n📋 PARAMETERS:")
                for k, v in results["params"].items():
                    print(f"  {k}: {v}")
                
                print("\n⏱️  TIMING:")
                for step, data in results["steps"].items():
                    print(f"  {step}: {data.get('time', 'N/A')}s")
                print(f"  TOTAL: {results['total_time_sec']}s")
                
                print("\n📊 RESULTS:")
                for k, v in results["steps"].items():
                    if k == "filter":
                        print(f"  {k}:")
                        print(f"    - Total evaluated: {v.get('total_evaluated')}")
                        print(f"    - Selected: {v.get('selected')}")
                        print(f"    - Top scores: {v.get('top_scores')}")
                    elif k == "transcribe":
                        print(f"  {k}: {v.get('language')} ({v.get('language_prob')})")
                    elif k == "windows":
                        print(f"  {k}: {v.get('count')} windows")
                    elif k == "evaluate":
                        print(f"  {k}: {v.get('count')} evaluations")
                    elif k == "render":
                        print(f"  {k}: {v.get('count')} shorts")
                
                print("\n🎬 SHORTS GENERATED:")
                for short in results["outputs"].get("shorts", []):
                    print(f"  #{short['rank']}: {short['title']} (score: {short['score']}, {short['size_mb']}MB)")
                    print(f"      → {short['output_path']}")
            else:
                print(f"\n❌ ERROR: {results.get('error')}")
            
            print("\n" + "="*60)
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
