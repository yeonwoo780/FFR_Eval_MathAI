import nlptutti as metrics
# https://github.com/hyeonsangjeon/computing-Korean-STT-error-rates
from datasets import load_dataset, Dataset
import soundfile as sf
import io
import base64
import requests
import torch
import re
import pandas as pd

def count_characters(reference: str, include_space=True) -> int:
    """
    CER 계산용 N (문자 수)
    include_space = True -> 공백 포함
    """
    if include_space:
        return len(reference)
    else:
        return len(reference.replace(" ", ""))

def replace_sentence(reference):
    """
    원본 Text 전처리. ', " 삭제 후 양 옆 공백 삭제
    """
    reference = reference.replace("'","").replace('"',"")
    reference = reference.strip()
    return reference

def strip_special_chars(reference):
    "특수문자 삭제"
    reference = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', reference)
    return reference

if __name__ =="__main__":
    Dataset.cleanup_cache_files
    dname = "google/fleurs"
    dlang = "korean"
    print(f"Load Dataset: {dname}\nLanguage: {dlang}")

    dataset = load_dataset(dname, "ko_kr", split="train", streaming=True, trust_remote_code=True)

    seed = 42
    print(f"Set Dataset seed: {seed}")

    tsetnum = 100
    test_datasets = dataset.shuffle(seed=seed).take(tsetnum)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    Eval_list = []
    n = 0
    for test_dataset in test_datasets:
        audio_info = test_dataset['audio']
        language = "korean"
        file_path = audio_info['path']
        reference = test_dataset['transcription']
        
        audio_array = audio_info['array']
        sampling_rate = audio_info['sampling_rate']

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_array, sampling_rate, format='wav')
        wav_buffer.seek(0)
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')

        url = "http://192.168.1.158:8888/wer"
        payload = {
            "audio_base64": audio_base64,
        }
        response = requests.post(url, json=payload)
        preds = response.text.strip()
        reference = replace_sentence(reference)
        reference = strip_special_chars(reference)
        preds = replace_sentence(preds)
        preds = strip_special_chars(preds)
        result = metrics.get_cer(reference, preds, rm_punctuation=True)
        cer = round(result['cer'],5)
        substitutions = result['substitutions']
        insertions = result['insertions']
        deletions = result['deletions']
        N = count_characters(reference, False)
        n += 1
        Eval_list.append({
            'num': n,
            'Dataset': dname,
            'file_path': file_path,
            'language': language,
            'reference': reference,
            'prediction': preds,
            'S' : substitutions,
            'I': insertions,
            'D': deletions,
            'N': N,
            'CER': cer
        })
        print(f"total: {tsetnum} 중 {n}번째 데이터 실행 완료")
        torch.cuda.empty_cache()

    df = pd.DataFrame(Eval_list)
    fpath = "cer_eval_korean.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    averages = round(df["CER"].mean(), 5)
    print(f"평균 CER : {averages*100}%")