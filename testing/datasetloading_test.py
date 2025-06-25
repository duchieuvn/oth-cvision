import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from utils import DynamicNucDataset, UsForKidneyDataset, Covid19RadioDataset  # import your dataset classes here

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def test_dataset(dataset_class, root, split, label):
    print(f"\n🔍 Testing {label}: {split} set")
    try:
        dataset = dataset_class(root=root, subset=split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        sample = next(iter(loader))
        print(f"✅ Loaded {label} [{split}]: {len(dataset)} samples, first sample shape: {sample[0].shape}")
    except Exception as e:
        print(f"❌ Failed to load {label} [{split}]: {e}")

if __name__ == "__main__":
    config = load_config()

    # DynamicNuclear
    test_dataset(DynamicNucDataset, config['data_root_dn'], config['train_dn'], 'DynamicNuclear')
    test_dataset(DynamicNucDataset, config['data_root_dn'], config['val_dn'], 'DynamicNuclear')

    # UsForKidney
    test_dataset(UsForKidneyDataset, config['data_root_ufk'], config['train_ufk'], 'UsForKidney')
    test_dataset(UsForKidneyDataset, config['data_root_ufk'], config['val_ufk'], 'UsForKidney')

    # Covid19Radio (global split)
    test_dataset(Covid19RadioDataset, config['data_root_covid'], config['train_covid'], 'Covid19Radio')
    test_dataset(Covid19RadioDataset, config['data_root_covid'], config['val_covid'], 'Covid19Radio')

    # Optional: test category-specific splits
    for cat in ['C1', 'C2', 'C3', 'C4']:
        subset = f"train_{cat}"
        test_dataset(Covid19RadioDataset, config['data_root_covid'], subset, f"Covid19Radio-{cat}")
