# Example API Requests

Start the server first:

```bash
uvicorn app.main:app --reload --port 8000
```

## 1. General HFpEF education question

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is HFpEF and what symptoms does it cause?"}'
```

## 2. Treatment-related question

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I ask my doctor about HFpEF treatment options?"}'
```

## 3. Question with insufficient evidence (refused safely)

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the best brand of running shoes for a marathon?"}'
```

## 4. High-risk symptom (guardrail escalation)

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "I am having severe chest pain and shortness of breath right now."}'
```

## 5. Vague / ambiguous question (refused safely)

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about it."}'
```

## Service info / health check

```bash
curl -s http://127.0.0.1:8000/
```

Full structured responses for all five cases are saved in
`sample_responses.json` in this folder.
