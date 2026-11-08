"""Full-set identity, deterministic samples, and resumable output protocol."""
import hashlib
import json
from pathlib import Path

PROMPT_SHA256 = '5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13'
METHODS = ['janus_baseline', 'janus_inchworm', 'llamagen_baseline']
TASKS = ['single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path):
    if sha(path) != PROMPT_SHA256:
        raise ValueError('Official full prompt set hash mismatch')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 553 or {r['tag'] for r in rows} != set(TASKS):
        raise ValueError('Expected official 553 prompts and six tasks')
    return rows


def select_ids(rows, scope):
    if scope == 'full':
        return list(range(len(rows)))
    if scope != 'pilot':
        raise ValueError(scope)
    return [next(i for i, row in enumerate(rows) if row['tag'] == tag) for tag in TASKS]


def seed_for(prompt_id, sample_id):
    return 42 + 4 * prompt_id + sample_id


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def bind_run(out, specification):
    out = Path(out)
    record = out/'run_spec.json'
    if record.exists():
        if json.loads(record.read_text(encoding='utf-8')) != specification:
            raise ValueError('Existing run has different source/config/prompt selection; use a new directory')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Refusing to adopt a nonempty directory without run_spec.json')
        atomic_json(record, specification)


def paths(out, method, prompt_id, sample_id):
    folder = Path(out)/method/f'{prompt_id:05d}'
    return folder/'samples'/f'{sample_id:04d}.png', folder/'records'/f'{sample_id:04d}.json'


def completed(image, receipt, specification, row):
    if not Path(image).exists() or not Path(receipt).exists():
        return False
    data = json.loads(Path(receipt).read_text(encoding='utf-8'))
    if data['specification'] != specification or data['metadata'] != row or data['image_sha256'] != sha(image):
        raise ValueError(f'Existing sample does not match its receipt: {image}')
    return True


def commit_image(out, method, prompt_id, sample_id, row, specification, image, diagnostics):
    target, receipt = paths(out, method, prompt_id, sample_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = target.parent.parent/'metadata.jsonl'
    if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8')) != row:
        raise ValueError(f'Metadata conflict: {metadata}')
    atomic_json(metadata, row)
    temp = target.with_name(target.name + '.part')
    image.save(temp, format='PNG')
    temp.replace(target)
    atomic_json(receipt, {'specification': specification, 'metadata': row,
                         'image_sha256': sha(target), **diagnostics})


def validate_outputs(out, method, rows, ids, samples):
    root = Path(out)/method
    spec_path = Path(out)/'run_spec.json'
    expected_spec = json.loads(spec_path.read_text(encoding='utf-8')) if spec_path.exists() else None
    expected_ids = {f'{i:05d}' for i in ids}
    actual_ids = {p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()}
    if actual_ids != expected_ids:
        raise ValueError(f'{method}: missing or extra prompt directories')
    expected_samples = {f'{j:04d}.png' for j in range(samples)}
    for i in ids:
        folder = root/f'{i:05d}'
        if json.loads((folder/'metadata.jsonl').read_text(encoding='utf-8')) != rows[i]:
            raise ValueError(f'Metadata mismatch for prompt {i}')
        if {p.name for p in (folder/'samples').iterdir() if p.suffix == '.png'} != expected_samples:
            raise ValueError(f'Incomplete or extra samples for prompt {i}')
        for j in range(samples):
            image, receipt = paths(out, method, i, j)
            if not receipt.is_file():
                raise ValueError(f'Missing receipt: {image}')
            record = json.loads(receipt.read_text(encoding='utf-8'))
            if (record['image_sha256'] != sha(image) or record['metadata'] != rows[i]
                or (expected_spec is not None and record['specification'] != expected_spec)):
                raise ValueError(f'Missing or mismatched receipt: {image}')
    return len(ids) * samples
