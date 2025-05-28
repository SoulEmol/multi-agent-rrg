import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from evaluate import load
from bert_score import score as bert_score_fn

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    refs = [x["original_report"].strip() for x in data]
    preds = [x["image_caption"].strip() for x in data]
    return refs, preds

def compute_bleu(refs, preds):
    refs_tok = [[ref.split()] for ref in refs]
    preds_tok = [pred.split() for pred in preds]
    bleu = corpus_bleu(refs_tok, preds_tok, smoothing_function=SmoothingFunction().method1)
    return bleu

def compute_rouge(refs, preds):
    rouge = load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_aggregator=True)
    return scores

def compute_meteor(refs, preds):
    meteor = load("meteor")
    score = meteor.compute(predictions=preds, references=refs)
    return score["meteor"]

def compute_bertscore(refs, preds):
    P, R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item()
    }

if __name__ == "__main__":
    path = "/mnt/storage/home/zy0058@students.ad.unt.edu/mywork/outputs_full_3/synthesis/synthesized_reports.json"
    refs, preds = load_data(path)

    print("🔹 BLEU Score:")
    print(f"  BLEU-4: {compute_bleu(refs, preds):.4f}")

    print("\n🔹 ROUGE Scores:")
    rouge_scores = compute_rouge(refs, preds)
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")

    print("\n🔹 METEOR Score:")
    print(f"  METEOR: {compute_meteor(refs, preds):.4f}")

    print("\n🔹 BERTScore:")
    bs = compute_bertscore(refs, preds)
    for k, v in bs.items():
        print(f"  {k}: {v:.4f}")
