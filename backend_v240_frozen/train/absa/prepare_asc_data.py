import json
from collections import Counter
from pathlib import Path
from config_asc import SOURCE_DATA_DIR, ASC_DATA_DIR

SOURCE_FILES = {
    'train': SOURCE_DATA_DIR / 'train.json',
    'validation': SOURCE_DATA_DIR / 'validation.json',
    'test': SOURCE_DATA_DIR / 'test.json',
}
VALID_POLARITIES = {'NEGATIVE', 'NEUTRAL', 'POSITIVE'}

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def convert_split(rows, split_name):
    output = []
    seen_instance_ids = set()
    for row in rows:
        comment_id = str(row['comment_id'])
        video_id = str(row['video_id'])
        text = str(row['text'])
        for aspect_index, aspect in enumerate(row.get('aspects') or []):
            term = str(aspect.get('term') or '')
            start = aspect.get('start')
            end = aspect.get('end')
            polarity = str(aspect.get('polarity') or '').upper()
            if polarity not in VALID_POLARITIES:
                raise ValueError(f'{split_name} / comment {comment_id}: polarity không hợp lệ: {polarity}')
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f'{split_name} / comment {comment_id}: start/end phải là int')
            if start < 0 or end > len(text) or start >= end:
                raise ValueError(f'{split_name} / comment {comment_id}: span [{start}, {end}] không hợp lệ')
            actual_term = text[start:end]
            if actual_term != term:
                raise ValueError(f"{split_name} / comment {comment_id}: term='{term}' != text[start:end]='{actual_term}'")
            instance_id = f'{comment_id}:{start}:{end}'
            if instance_id in seen_instance_ids:
                raise ValueError(f'Duplicate ASC instance: {instance_id}')
            seen_instance_ids.add(instance_id)
            output.append({
                'instance_id': instance_id,
                'comment_id': comment_id,
                'video_id': video_id,
                'text': text,
                'aspect': term,
                'aspect_start': start,
                'aspect_end': end,
                'label': polarity,
                'aspect_index': aspect_index,
            })
    return output

def split_stats(rows):
    labels = Counter(row['label'] for row in rows)
    return {
        'instances': len(rows),
        'comments_with_aspects': len({row['comment_id'] for row in rows}),
        'videos': len({row['video_id'] for row in rows}),
        'label_distribution': dict(labels),
    }

def main():
    print('=' * 70)
    print('PREPARE ASC DATASET')
    print('=' * 70)
    ASC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    video_sets = {}
    for split_name, source_path in SOURCE_FILES.items():
        source_rows = load_json(source_path)
        converted = convert_split(source_rows, split_name)
        output_path = ASC_DATA_DIR / ('validation.json' if split_name == 'validation' else f'{split_name}.json')
        save_json(output_path, converted)
        all_stats[split_name] = split_stats(converted)
        video_sets[split_name] = {row['video_id'] for row in converted}
        print(f"{split_name}: {len(converted)} ASC instances -> {all_stats[split_name]['label_distribution']}")
    if video_sets['train'] & video_sets['validation']:
        raise ValueError('Video leakage: train ↔ validation')
    if video_sets['train'] & video_sets['test']:
        raise ValueError('Video leakage: train ↔ test')
    if video_sets['validation'] & video_sets['test']:
        raise ValueError('Video leakage: validation ↔ test')
    save_json(ASC_DATA_DIR / 'dataset_stats.json', {
        'split_strategy': 'reuse_fixed_video_split_from_ATE',
        **all_stats,
    })
    print('[OK] Không có video leakage.')
    print('[OK] ASC dataset saved to:', ASC_DATA_DIR)

if __name__ == '__main__':
    main()
