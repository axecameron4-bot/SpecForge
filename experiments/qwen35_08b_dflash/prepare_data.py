"""Download pinned, seeded chat subsets and a separate GSM8K benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-size', type=int, default=8192)
    p.add_argument('--validation-size', type=int, default=256)
    p.add_argument('--benchmark-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--refresh-revisions', action='store_true', help='Resolve current Hub revisions instead of the recorded ones')
    a = p.parse_args()
    out = Path('cache/dataset/qwen35_08b_dflash')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('experiments/qwen35_08b_dflash/results/data_manifest.json')
    pinned = json.loads(manifest_path.read_text())['datasets'] if manifest_path.exists() and not a.refresh_revisions else {}
    manifest = {'seed': a.seed, 'files': {}, 'datasets': dict(pinned)}
    api = HfApi()
    for repo, split, size, name, config in [
        ('HuggingFaceH4/ultrachat_200k', 'train_sft', a.train_size, 'train', None),
        ('HuggingFaceH4/ultrachat_200k', 'test_sft', a.validation_size, 'validation', None),
        ('openai/gsm8k', 'test', a.benchmark_size, 'gsm8k', 'main'),
    ]:
        if repo not in manifest['datasets']:
            manifest['datasets'][repo] = api.dataset_info(repo).sha
        revision = manifest['datasets'][repo]
        ds = load_dataset(repo, config, split=split, revision=revision,
                          cache_dir='cache/huggingface/datasets')
        ds = ds.shuffle(seed=a.seed).select(range(min(size, len(ds))))
        rows = []
        for row in ds:
            if name == 'gsm8k':
                record = {'id': hashlib.sha256(row['question'].encode()).hexdigest(),
                          'question': row['question'], 'answer': row['answer']}
            else:
                # First turn only makes assistant span masking unambiguous.
                messages = row['messages'][:2]
                if [m['role'] for m in messages] != ['user', 'assistant']:
                    continue
                record = {'id': row['prompt_id'], 'conversations': messages}
            rows.append(record)
        path = out / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        manifest['files'][name] = {'path': str(path), 'count': len(rows), 'split': split,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, len(rows), flush=True)
    train_ids = {json.loads(l)['id'] for l in (out/'train.jsonl').read_text().splitlines()}
    val_ids = {json.loads(l)['id'] for l in (out/'validation.jsonl').read_text().splitlines()}
    if train_ids & val_ids:
        raise ValueError('Training / validation prompt ID overlap')
    manifest_path.write_text(
        json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
