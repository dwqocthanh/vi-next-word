import argparse
from pathlib import Path
 
import sentencepiece as spm
 
from src.config import PRETRAINED, PROC_DIR, SPM_PREFIX
 
 
def train_spm(vocab_size=8000):
    Path("models").mkdir(exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=f"{PROC_DIR}/train.txt",
        model_prefix=SPM_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,          # giữ đủ ký tự có dấu tiếng Việt
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
 
 
def fertility(encode, lines):
    """Số token trung bình cho mỗi âm tiết (tách bằng khoảng trắng).
    Càng thấp = tokenizer càng 'hiểu' tiếng Việt."""
    n_tok = n_syl = 0
    for line in lines:
        n_syl += len(line.split())
        n_tok += len(encode(line))
    return n_tok / n_syl
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--hf", action="store_true",
                    help="so sánh thêm với tokenizer của mô hình pretrained")
    args = ap.parse_args()
 
    if args.train:
        train_spm()
 
    with open(f"{PROC_DIR}/val.txt", encoding="utf-8") as f:
        lines = [l.strip() for l in f][:2000]
 
    sp = spm.SentencePieceProcessor(model_file=SPM_PREFIX + ".model")
    # Kiểm tra mã hoá rồi giải mã ra lại câu cũ
    s = "Hôm nay trời đẹp"
    print("Round-trip        :", sp.decode(sp.encode(s)) == s)   # mong đợi True
    
    print("Ví dụ SentencePiece:", sp.encode("Hôm nay trời đẹp", out_type=str))
    print("SentencePiece 8k  :", round(fertility(sp.encode, lines), 3),
          "token/âm tiết")
    print("Theo khoảng trắng :", 1.0, "token/âm tiết (mốc so sánh)")
 
    if args.hf:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(PRETRAINED)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        print("Ví dụ GPT-2 VN    :", tok.tokenize("Hôm nay trời đẹp"))
        print(f"{PRETRAINED}:", round(fertility(enc, lines), 3),
              "token/âm tiết")
        print("Kích thước từ điển:", len(tok))
 
if __name__ == "__main__":
    main()
