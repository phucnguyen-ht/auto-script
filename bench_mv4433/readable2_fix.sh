#!/bin/bash
CURL_CMD="curl -s http://localhost:9001/v1/completions -H \"Content-Type: application/json\""
# MODEL_PATH=/app/vast0/share-mv/LGAI-EXAONE/EXAONE-3.5-32B-Instruct
# MODEL_PATH=/app/vast0/share-mv/deepseek-ai/DeepSeek-V3.2
# MODEL_PATH=/app/share-mv/openai/gpt-oss-120b
# MODEL_PATH=/app/share-mv/deepseek-ai/DeepSeek-R1

MODEL_PATH=/remote/vast0/share-mv/zai-org/GLM-5-FP8/
# MODEL_PATH=/app/vast0/share-mv/lmsys/gpt-oss-120b-bf16
# MODEL_PATH=/remote/vast0/share-mv/anhduong/models/step3

make_requests() {
    local prompt="$1"
    echo "Making requests for prompt: $prompt"
    echo "------------------------------------------"

    response=$(eval "$CURL_CMD -d '{
        \"model\": \"$MODEL_PATH\",
        \"prompt\": \"$prompt\",
        \"max_tokens\": 300,
        \"temperature\": 0
    }'")
    echo "$response" | sed -n 's/.*"text":\(.*\),"logprobs".*/\1/p'
    # echo "$response"
    echo ""
    echo ""
}

make_requests "Who won the world series in 2020?"
# make_requests "Who won the world series in 2020?"
# make_requests "What are the main causes of climate change?"
# make_requests "Can you summarize the plot of Pride and Prejudice?"
# make_requests "What are the health benefits of regular exercise?"
# make_requests "How does photosynthesis work in plants?"
# make_requests "What are the key themes in Shakespeare Hamlet?"
# make_requests "What is the capital of France?"
# make_requests "Who painted the Mona Lisa?"
# make_requests "What is the largest planet in our solar system?"
# make_requests "What are the primary colors?"
# make_requests "What is the chemical symbol for water?"
# make_requests "Who wrote Romeo and Juliet?"
# make_requests "What is the speed of light?"
# make_requests "What is the tallest mountain in the world?"
# make_requests "What is the currency of Japan?"
# make_requests "What is the definition of a black hole?"