from together_provider import call_together

print(call_together(
    model="deepseek-ai/DeepSeek-V4-Pro",
    max_output_tokens=80,
    temperature=0.6,
    system_prompt="Sei un esperto di scienze naturali.",
    user_prompt="Quale pianeta è conosciuto come il pianeta rosso",
    reasoning=False,
    api_key=""
))
