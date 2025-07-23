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
