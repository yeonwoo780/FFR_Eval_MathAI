from datasets import load_dataset, Dataset
import torch
import re
import requests
import pandas as pd

def contains_korean(text):
    # 자모 + 호환 자모 + 완성형 인지 확인
    # 위 조건에 부합하면 1 부합하지 않으면 0
    pattern = r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]'  
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
            "question": en_quest,
            "language": "korean"
        }
        response = requests.post(url, json=payload)
        context = response.text
        Score = contains_korean(context)
        n += 1
        Eval_list.append({
            'num': n,
            '원본 언어(입력값)': en_quest,
            '원본 답변': ko_quest,
            'sLLM 한국어 답변': context,
            'Score': Score
        })
        if Score == 0:
            A+=1
        print(f"total: {B} 중 {n}번째 데이터 실행 완료")
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(Eval_list)
    fpath = "sllm_eval_korean.csv"
    df.to_csv(fpath, encoding="utf-8-sig", index=False)
    print("기본 Score 산출식: 1 - (A/B)")
    print(f"한국어 결과를 제공하지 않은 데이터 수: {A}")
    print(f"전체 데이터 수: {B}")
    print(f"Score 산출식: 1 - ({A}/{B}) ")
    TotalScore = (1 - (A/B))*100
    print(f"결과: {TotalScore} %")