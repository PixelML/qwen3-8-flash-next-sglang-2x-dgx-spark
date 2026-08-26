# API integration

Rank zero serves an OpenAI-compatible API. Keep it on a private network or
publish it through an authenticated TLS reverse proxy.

## Verify the model list

```bash
export QWEN_BASE_URL="https://your-proxy.example/v1"
export QWEN_API_KEY="replace-me"

curl -fsS \
  -H "Authorization: Bearer ${QWEN_API_KEY}" \
  "${QWEN_BASE_URL}/models"
```

## Coding request

```bash
curl -fsS "${QWEN_BASE_URL}/chat/completions" \
  -H "Authorization: Bearer ${QWEN_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Implement binary search in Python."}],
    "reasoning_effort": "high",
    "max_tokens": 512
  }'
```

Declare both `text` and `image` as input modalities in clients that require an
explicit model registry. The endpoint supports reasoning content, tool calls,
and image input.
