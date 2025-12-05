from datasets import load_dataset, Dataset
import torch
import re
import requests
import pandas as pd

def contains_english(text):
    pattern = r'[A-Za-z]'  # 영어 알파벳 범위
    return 1 if re.search(pattern, text) else 0

if __name__ =="__main__":
    Dataset.cleanup_cache_files
    dname = "heegyu/OIG-small-chip2-ko"
    dataset = load_dataset(dname, split="train", streaming=True)

    seed = 42
    print(f"Set Dataset seed: {seed}")

    B = 200
    print(f"Get Test Dataset: {B}")
    test_datasets = dataset.shuffle(seed=seed).take(B)

    Eval_list = []
    n = 0
    A = 0
    for i in test_datasets:
        en_quest = i['user']
        en_ans = i['chip2']
        ko_quest = i['user_translated']
        ko_ans = i['chip2_translated']
        url = "http://192.168.1.158:8888/sLLM"
        payload = {
            "question": ko_quest,
            "language": "english"
        }
        response = requests.post(url, json=payload)
        context = response.text
        Score = contains_english(context)
        n += 1
        Eval_list.append({
            'num': n,
            '원본 언어(입력값)': ko_quest,
            '원본 답변': en_quest,
            'sLLM 영어 답변': context,
            'Score': Score
        })
        if Score == 0:
            A+=1
        print(f"total: {B} 중 {n}번째 데이터 실행 완료")
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(Eval_list)
    fpath = "sllm_eval_english.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    print("기본 Score 산출식: 1 - (A/B)")
    print(f"영어 결과를 제공하지 않은 데이터 수: {A}")
    print(f"전체 데이터 수: {B}")
    print(f"Score 산출식: 1 - ({A}/{B}) ")
    TotalScore = (1 - (A/B))*100
    print(f"결과: {TotalScore} %")