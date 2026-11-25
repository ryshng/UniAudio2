import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, default=Path('outputs/EmoMatchSpanishDBGenerated'))
parser.add_argument('--max-samples', type=int)
args, inference_args = parser.parse_known_args()

emotions = {
    'Alegria': 'happy',
    'Asco': 'disgusted',
    'Ira': 'angry',
    'Miedo': 'afraid',
    'Sorpresa': 'surprised',
    'Tristeza': 'sad',
}
script = Path(__file__).resolve().with_name('multi_task_inference.py')
args.output_dir.mkdir(parents=True, exist_ok=True)
audios = sorted(path for path in args.dataset_dir.rglob('*')
                if path.suffix.lower() == '.wav' and '_Neutro_' in path.name)

for audio in audios[:args.max_samples]:
    speaker, _, sentence = audio.stem.split('_')
    output = args.output_dir / audio.stem
    for emotion, style in emotions.items():
        print(f'{audio.name} -> {emotion}', flush=True)
        subprocess.run([
            sys.executable, str(script),
            '--prompt_text', 'Edit the speech according to the following instruction:',
            *inference_args,
            '--task', 'speech_edit', '--stage', 'all',
            '--audio', str(audio), '--text', f'Make the speech sound {style}',
            '--output_dir', str(output),
        ], check=True)
        generated = output / 'wavs' / f'{audio.stem}.wav'
        generated.replace(output / 'wavs' / f'{speaker}_{emotion}_{sentence}_generated.wav')
