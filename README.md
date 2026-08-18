# skala-Transformer

SKALA Transformer 실습. Self-Attention을 NumPy 수준에서 직접 구현해보고, 이어서 문자 단위 미니 GPT를 처음부터 만들어 학습시킵니다.

## 노트북

### `실습_Self_Attention_Prototype.ipynb`
Scaled Dot-Product Self-Attention을 라이브러리 없이 단계별로 구현합니다.

정규식 토크나이저 → Vocabulary 생성 → 정수 인코딩 → Token Embedding → Q/K/V 투영 → Attention Weight 계산 → Context Vector 도출 순으로, 각 단계의 텐서 크기를 확인하며 진행합니다. 마지막에 Attention Weight를 히트맵으로 시각화하고, 특정 Query 토큰이 어느 토큰에 주목하는지 따로 확인합니다.

입력 코퍼스: `mini_corpus.txt`

### `종합실습_charGPT_transformer.ipynb`
김유정 단편소설을 학습 데이터로 문자 단위 GPT를 밑바닥부터 구현합니다.

- 문자 토큰화와 `CharDataset`
- `GPTConfig` 모델 설정
- Self-Attention + Causal Mask
- Transformer Block (Residual Connection · LayerNorm · MLP)
- Embedding → Logits → Loss로 이어지는 전체 모델
- Trainer 구성 및 학습

학습 전 / 중간 / 후 생성 결과를 나란히 비교해, 모델이 문자 조합부터 단어·문장 형태를 갖춰가는 과정을 관찰합니다.

입력 코퍼스: `input_kr.txt`

## 실행

```bash
pip install torch numpy matplotlib
jupyter notebook
```
