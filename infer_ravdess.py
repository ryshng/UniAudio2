import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, default=Path('outputs/RAVDESSGenerated'))
parser.add_argument('--max-samples', type=int)
args, inference_args = parser.parse_known_args()

emotions = {
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'afraid',
    '07': 'disgusted',
    '08': 'surprised',
}

script = Path(__file__).resolve().with_name('multi_task_inference.py')
args.output_dir.mkdir(parents=True, exist_ok=True)

audios = sorted(
    path for path in args.dataset_dir.rglob('03-01-01-*.wav')
)

for audio in audios[:args.max_samples]:
    modality, channel, _, intensity, sentence, repetition, actor = audio.stem.split('-')
    output = args.output_dir / audio.stem

    for emotion, style in emotions.items():
        print(f'{audio.name} -> {style}', flush=True)

        subprocess.run([
            sys.executable, str(script),
            '--prompt_text', 'Edit the speech according to the following instruction:',
            *inference_args,
            '--task', 'speech_edit', '--stage', 'all',
            '--audio', str(audio),
            '--text', f'Make the speech sound {style}',
            '--output_dir', str(output),
        ], check=True)

        generated = output / 'wavs' / f'{audio.stem}.wav'

        filename = (
            f'{modality}-{channel}-{emotion}-{intensity}-'
            f'{sentence}-{repetition}-{actor}_generated.wav'
        )

        generated.replace(output / 'wavs' / filename)