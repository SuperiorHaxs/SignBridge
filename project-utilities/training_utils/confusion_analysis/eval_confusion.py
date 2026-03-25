"""Evaluate a production model and print confusion pairs + per-class accuracy."""
import sys, os, json, torch, argparse
import numpy as np
from pathlib import Path

# Resolve project root (three levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'openhands-modernized' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'models' / 'training-scripts'))

from util.openhands_modernized_inference import load_model_from_checkpoint
from train_asl import WLASLOpenHandsDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate model confusion pairs")
    parser.add_argument("--model-dir", required=True, help="Path to production model directory")
    parser.add_argument("--manifest-dir", required=True, help="Path to split manifests directory")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate")
    parser.add_argument("--top-n", type=int, default=25, help="Number of top confusion pairs to show")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest_dir = Path(args.manifest_dir)

    # Load model
    model, id_to_gloss, masked_ids = load_model_from_checkpoint(str(model_dir))
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load class mapping
    with open(model_dir / 'class_index_mapping.json') as f:
        id_to_gloss = json.load(f)
    gloss_to_id = {v: int(k) for k, v in id_to_gloss.items()}
    num_classes = len(id_to_gloss)

    # Load manifest
    manifest_file = manifest_dir / f'{args.split}_manifest.json'
    with open(manifest_file) as f:
        manifest = json.load(f)

    # Resolve pickle pool - use project-root-relative path
    pickle_pool = PROJECT_ROOT / 'datasets' / 'augmented_pool' / 'pickle'

    val_files = []
    val_labels = []
    for gloss, families in manifest['classes'].items():
        for family in families:
            for fname in family['files']:
                fpath = pickle_pool / gloss / fname
                if fpath.exists():
                    val_files.append(str(fpath))
                    val_labels.append(gloss)

    print(f'{args.split.title()} files found: {len(val_files)}')
    print(f'Unique glosses: {len(set(val_labels))}')

    if not val_files:
        print("ERROR: No files found. Check pickle pool path.")
        return 1

    # Load model config to match feature extraction
    config_path = model_dir / 'config.json'
    with open(config_path) as f:
        model_config = json.load(f)

    use_motion = model_config.get('use_motion_features', False)
    use_spatial = model_config.get('use_spatial_features', False)
    expected_spatial_dim = model_config.get('spatial_features', 26)
    expected_features = model_config.get('pose_features', 279)

    print(f'Feature config: motion={use_motion}, spatial={use_spatial} (dim={expected_spatial_dim})')
    print(f'Expected pose_features: {expected_features}')

    # Create dataset and loader
    MAX_SEQ_LENGTH = 256
    dataset = WLASLOpenHandsDataset(
        val_files, val_labels, gloss_to_id, MAX_SEQ_LENGTH,
        augment=False, use_finger_features=True,
        use_motion_features=use_motion, use_spatial_features=use_spatial,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Check if spatial features need truncation (code may produce more features than model expects)
    truncate_spatial = False
    if use_spatial:
        current_spatial = dataset.processor.extract_spatial_features(
            np.zeros((10, 83, 3), dtype=np.float32)
        ).shape[-1]
        if current_spatial != expected_spatial_dim:
            truncate_spatial = True
            print(f'WARNING: Spatial features mismatch: code produces {current_spatial}, model expects {expected_spatial_dim}. Will truncate.')

    # Run inference
    all_preds = []
    all_labels = []
    top3_per_class = {}  # class_id -> [top3_correct, total]

    with torch.no_grad():
        for batch in loader:
            pose_sequences = batch['pose_sequence'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            finger_features = batch.get('finger_features')
            if finger_features is not None:
                finger_features = finger_features.to(device)
            motion_features = batch.get('motion_features')
            if motion_features is not None:
                motion_features = motion_features.to(device)
            spatial_features = batch.get('spatial_features')
            if spatial_features is not None:
                if truncate_spatial:
                    spatial_features = spatial_features[:, :, :expected_spatial_dim]
                spatial_features = spatial_features.to(device)
            logits = model(pose_sequences, attention_masks, finger_features, motion_features, spatial_features)
            _, predicted = torch.max(logits, 1)
            _, top3_predicted = torch.topk(logits, 3, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(len(labels)):
                label_val = labels[i].item()
                if label_val not in top3_per_class:
                    top3_per_class[label_val] = [0, 0]  # [correct, total]
                top3_per_class[label_val][1] += 1
                if label_val in top3_predicted[i].cpu().numpy():
                    top3_per_class[label_val][0] += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    # Find top confusion pairs
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i][j] > 0:
                pairs.append((confusion[i][j], id_to_gloss[str(i)], id_to_gloss[str(j)]))

    pairs.sort(reverse=True)

    overall_acc = (all_preds == all_labels).sum() / len(all_labels) * 100
    print(f'\nOverall {args.split.title()} Accuracy: {overall_acc:.2f}%')

    print(f'\n=== TOP {args.top_n} CONFUSION PAIRS ===')
    print(f'Count  True Label       ->  Predicted As')
    print('-' * 50)
    for count, true_g, pred_g in pairs[:args.top_n]:
        print(f'{count:>5}  {true_g:<16} ->  {pred_g:<16}')

    # Overall top-3
    total_top3_correct = sum(v[0] for v in top3_per_class.values())
    total_top3_total = sum(v[1] for v in top3_per_class.values())
    overall_top3 = total_top3_correct / total_top3_total * 100 if total_top3_total > 0 else 0
    print(f'Overall {args.split.title()} Top-3 Accuracy: {overall_top3:.2f}%')

    # Per-class accuracy (worst first)
    print(f'\n=== PER-CLASS ACCURACY (worst to best) ===')
    print(f'Gloss            Correct  Total  Top1%  Top3%')
    print('-' * 55)
    class_accs = []
    for i in range(num_classes):
        total = confusion[i].sum()
        correct = confusion[i][i]
        acc = (correct / total * 100) if total > 0 else 0
        t3 = top3_per_class.get(i, [0, 0])
        top3_acc = (t3[0] / t3[1] * 100) if t3[1] > 0 else 0
        class_accs.append((acc, id_to_gloss[str(i)], correct, total, top3_acc))

    class_accs.sort()
    for acc, gloss, correct, total, top3_acc in class_accs:
        print(f'{gloss:<16} {correct:>7}  {total:>5}  {acc:>5.1f}%  {top3_acc:>5.1f}%')

    # Threshold summary
    top1_70 = sum(1 for a in class_accs if a[0] >= 70)
    top3_85 = sum(1 for a in class_accs if a[4] >= 85)
    print(f'\n=== THRESHOLD SUMMARY ===')
    print(f'Glosses with Top-1 >= 70%: {top1_70}/{num_classes} ({top1_70/num_classes*100:.0f}%)')
    print(f'Glosses with Top-3 >= 85%: {top3_85}/{num_classes} ({top3_85/num_classes*100:.0f}%)')

    return 0


if __name__ == "__main__":
    sys.exit(main())
