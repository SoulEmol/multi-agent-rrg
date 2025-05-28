# topk_agent.py

import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import open_clip
from training.data import IUXrayDataset


def dict_collate(batch):
    return list(zip(*batch))


class TopKAgent:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config["model_name_or_path"],
            pretrained=self.config["checkpoint_path"]
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.config["model_name_or_path"])

        self._build_dataloaders()

    def _build_dataloaders(self):
        self.train_dataset = IUXrayDataset(
            self.config["img_root"],
            self.config["db_json"],
            transforms=self.preprocess,
            tokenizer=self.tokenizer,
            load_include_path=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        self.eval_dataset = IUXrayDataset(
            self.config["img_root"],
            self.config["test_json"],
            transforms=self.preprocess,
            tokenizer=self.tokenizer,
            load_include_path=True,
            load_include_k=True,
            retrieval_k=int(self.config.get("fixed_k", 1))
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=dict_collate
        )

    def _get_logits(self, image_feats, text_feats, logit_scale):
        image_feats = image_feats.to(self.device)
        text_feats = text_feats.to(self.device)
        logits_per_image = (logit_scale * image_feats @ text_feats.T).cpu()
        return {
            "image_to_text": logits_per_image,
            "text_to_image": logits_per_image.T
        }

    def _retrieve_topk(self, logits, k_list, threshold=""):
        pred_list = []
        threshold = float(threshold) if threshold else None

        for i, k in enumerate(k_list):
            if k == 0:
                pred_list.append(torch.tensor([-1]))
                continue

            sims = logits["image_to_text"][i]
            sorted_vals, sorted_idx = sims.sort(descending=True)

            if threshold is not None:
                top1 = sorted_vals[0]
                ratios = top1 / sorted_vals
                selected = sorted_idx[ratios < threshold]
                if len(selected) > k:
                    selected = selected[:k]
                pred_list.append(selected)
            else:
                pred_list.append(sorted_idx[:k])

        return pred_list

    def run(self):
        print("🔍 开始提取训练文本特征...")
        train_text_features = []
        train_paths = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, texts, paths in tqdm(self.train_loader):
                images = images.to(self.device, dtype=self.dtype)
                texts = texts.to(self.device)
                feats = self.model(images, texts)[1]
                train_text_features.append(feats.cpu())
                train_paths.extend(paths)
            train_text_features = torch.cat(train_text_features)
            logit_scale = self.model.logit_scale.exp().mean()

        print("🖼️ 开始提取验证图像特征...")
        val_image_features, val_k_list, val_data_infos = [], [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, texts, paths, ks, infos in tqdm(self.eval_loader):
                images = torch.stack(images).to(self.device, dtype=self.dtype)
                texts = torch.stack(texts).to(self.device)

                feats = self.model(images, texts)[0]
                val_image_features.append(feats.cpu())
                val_k_list.extend([int(k) for k in ks])
                val_data_infos.extend(infos)
            val_image_features = torch.cat(val_image_features)

        print("📊 计算图文相似度并进行 Top-K 检索...")
        logits = self._get_logits(val_image_features, train_text_features, logit_scale)
        preds = self._retrieve_topk(logits, val_k_list, self.config.get("clip_threshold", ""))

        output_path = self.config["output_path"]
        print(f"📝 写入 Top-K 结果到：{output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            for pred_indices, data_info, actual_k in zip(preds, val_data_infos, [len(p) for p in preds]):
                references = []
                for idx in pred_indices:
                    if idx.item() == -1:
                        break
                    references.append(self.train_dataset.image_report_pairs[idx.item()][1])

                # 重命名和排序字段
                data_info.pop("id", None)
                image_id = data_info.pop("image_path", "")
                original_report = data_info.pop("report", "")
                split = data_info.get("split", "")

                formatted_entry = {
                    "image_id": image_id,
                    "split": split,
                    "original_report": original_report,
                    "reference_reports": references,
                    "retrieve_k": actual_k
                }

                f.write(json.dumps(formatted_entry) + "\n")

        print("✅ Top-K 检索完成！")

