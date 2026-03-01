import random
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOClassificationDataset(Dataset):
    def __init__(self, ann_file, img_dir, cls_names, transform=None, max_per_cls=None):
        self.coco      = COCO(str(ann_file))
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.name2idx  = {n: i for i, n in enumerate(cls_names)}

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.id2local = {c["id"]: self.name2idx[c["name"]] for c in cats if c["name"] in self.name2idx}
        target_ids    = list(self.id2local.keys())

        img_ids = set()
        for cid in target_ids:
            img_ids.update(self.coco.getImgIds(catIds=[cid]))

        buckets = defaultdict(list)
        for iid in img_ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, catIds=target_ids))
            c    = Counter(self.id2local[a["category_id"]] for a in anns)
            if c:
                buckets[c.most_common(1)[0][0]].append(iid)

        self.samples = []
        for label, ids in buckets.items():
            random.shuffle(ids)
            self.samples.extend((i, label) for i in (ids[:max_per_cls] if max_per_cls else ids))
        random.shuffle(self.samples)
        print(f"[dataset] {len(self.samples):,} images | {len(cls_names)} classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        iid, label = self.samples[idx]
        img = Image.open(self.img_dir / self.coco.loadImgs(iid)[0]["file_name"]).convert("RGB")
        return (self.transform(img) if self.transform else img), label
