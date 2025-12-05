from datasets import load_dataset, Dataset
from evaluate import load
import torch
from jiwer import compute_measures
import pandas as pd
import os
import re
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import soundfile as sf
import io
import base64
import requests

normalizer = BasicTextNormalizer()

wer_metric = load("wer")
def Eval(reference, prediction):
    wer = wer_metric.compute(references=[reference], predictions=[prediction])
    return wer

def english_normalize(text):
    norm_text = normalizer(text)
    return norm_text

def count_words(reference: str, include_punctuation=True) -> int:
    """
    WER 계산용 N (단어 수)
    include_punctuation = True -> 문장부호 포함한 단어 그대로 계산
    include_punctuation = False -> 문장부호 제거 후 단어 수 계산
    """
    if not include_punctuation:
        # 간단한 문장부호 제거
        import re
        reference = re.sub(r'[^\w\s]', '', reference)

    # 기본 WER: 공백 기준 단어 분리
    words = reference.split()

    return len(words)

if __name__ == "__main__":
    Dataset.cleanup_cache_files
    dname = "google/fleurs"
    dlang = "english"
    print(f"Load Dataset: {dname}\nLanguage: {dlang}")

    dataset = load_dataset(dname, "en_us", split="train", streaming=True, trust_remote_code=True)

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
        language = "english"
        file_path = audio_info['path']
        reference = test_dataset['transcription']
        reference = english_normalize(reference)

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
        prediction = english_normalize(preds)
        measures = compute_measures(reference, prediction)
        substitutions = measures['substitutions']
        insertions = measures['insertions']
        deletions = measures['deletions']
        N = count_words(reference)
        wer = Eval(reference, prediction)
        wer = round(wer,5)
        n += 1
        Eval_list.append({
            'num': n+1,
            'Dataset': dname,
            'language': language,
            'file_path': file_path,
            'reference': reference,
            'prediction': prediction,
            'S' : substitutions,
            'I' : insertions,
            'D' : deletions,
            'N' : N,
            'WER' : wer,
        })
        print(f"total: {tsetnum} 중 {n}번째 데이터 실행 완료")
        torch.cuda.empty_cache()

    df = pd.DataFrame(Eval_list)
    fpath = "wer_eval_english.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    averages = round(df["WER"].mean(), 5)
    print(f"평균 WER : {averages*100}%")