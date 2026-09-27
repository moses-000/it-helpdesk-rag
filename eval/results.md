# Evaluation results

44 test questions · embedding model `sentence-transformers/all-MiniLM-L6-v2`

## Retrieval

| Method | Hit@1 | Hit@3 | Hit@5 | Doc@3 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 50.0% | 75.0% | 79.5% | 86.4% | 0.620 | 0.1 |
| dense | 77.3% | 100.0% | 100.0% | 100.0% | 0.883 | 4.7 |
| hybrid | 63.6% | 90.9% | 95.5% | 97.7% | 0.767 | 4.6 |

## Answers (hybrid retrieval, top 4)

LLM: `openai/gpt-oss-20b`

| Metric | Score |
|---|---|
| Answer accuracy (key facts present) | 84.1% |
| Answers citing a source | 93.2% |
| Off-topic questions correctly declined | 100.0% |
