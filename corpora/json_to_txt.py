"""
This script aggregates transcripts from multiple Friends episode JSON files
into a single text file.

Functionality:
1. Reads all `.json` files from the `friends_episodes/` directory.
2. Iterates through each episode, scene, and utterance in the dataset.
3. Extracts:
    - `speakers`: joined by commas if multiple speakers are present.
    - `transcript`: the spoken line.
4. Formats each entry as: "<speakers>\t<transcript>".
5. Writes all lines to a single output file (`all_transcripts.txt`).

"""
import json
import glob

# Liste aller JSON-Dateien, z. B. alle .json im Ordner
json_files = glob.glob('friends_episodes/*.json')

output_lines = []

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for episode in data['episodes']:
        for scene in episode['scenes']:
            scene_id = scene['scene_id']
            for utterance in scene['utterances']:
                speakers = ', '.join(utterance['speakers'])
                transcript = utterance['transcript']
                line = f"{speakers}\t{transcript}"
                output_lines.append(line)

# Schreibe alle Zeilen in eine gemeinsame Datei
with open('all_transcripts.txt', 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"Done! Transcripts saved to all_transcripts.txt ({len(output_lines)} lines).")
