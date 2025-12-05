# 고장도수율 평가용

## UV 환경 설치

```bash
uv sync --extra cu126
```

## Eval

### version

**v0.2**

1. sLLM 모델 한국어 지원 확인

    ```bash
    uv run src/eval/v0.2/sllm_eval_korean.py
    ```

2. sLLM 모델 영어 지원 확인

    ```bash
    uv run src/eval/v0.2/sllm_eval_english.py
    ```

3. 다국어 음성인식 모델의 한국어 인식 오류율

    ```bash
    uv run src/eval/v0.2/cer_eval_korean.py
    ```

4. 다국어 음성인식 모델의 영어 인식 오류율

    ```bash
    uv run src/eval/v0.2/wer_eval_english.py
    ```