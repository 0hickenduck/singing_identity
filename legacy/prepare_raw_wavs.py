import os
import json
import subprocess

def main():
    manifest_path = "data/raw_audio/manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Channel mappings: channel_id -> folder_name
    channel_map = {
        "UCR6qhsLpn62WVxCBK1dkLow": "Enna",
        "UCL_qhgtOy0dy1Agp8vkySQg": "Mori",
        "UCHsx4Hqa-1ORjQTh9TYDhww": "Kiara",
        "UCt9H_RpQzhxzlyBxFqrdHqA": "FuwaMoco"
    }

    print(f"Loaded {len(manifest)} items from manifest.")

    for video_id, item in manifest.items():
        channel_id = item.get("channel_id")
        domain = item.get("domain")
        filename = item.get("filename")
        
        folder = channel_map.get(channel_id)
        if not folder:
            print(f"Skipping unknown channel_id {channel_id} for video {video_id}")
            continue

        # Map domain consistently
        if domain == "Speech":
            subfolder = "EN_JP_Talking" if folder in ["Mori", "FuwaMoco"] else "EN_Talking"
        elif domain == "Singing":
            subfolder = "EN_JP_Singing" if folder in ["Mori", "FuwaMoco"] else "EN_Singing"
        else:
            subfolder = domain

        src_path = os.path.join("data/raw_audio", filename)
        if not os.path.exists(src_path):
            print(f"Source file {src_path} not found. (Still downloading?)")
            continue

        dest_dir = os.path.join("data/raw_audio", folder, subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{video_id}.wav")

        if os.path.exists(dest_path):
            print(f"Destination {dest_path} already exists. Skipping.")
            continue

        print(f"Converting {src_path} -> {dest_path}...")
        cmd = [
            "ffmpeg", "-y",
            "-i", src_path,
            "-ar", "48000",
            "-ac", "2",
            dest_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Successfully converted {video_id} to WAV.")
        else:
            print(f"Failed to convert {video_id}: {res.stderr}")

if __name__ == "__main__":
    main()
