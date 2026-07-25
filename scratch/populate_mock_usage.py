import json
from pathlib import Path
from datetime import datetime, timedelta

usage_path = Path("a:/llama/llama.usage.json")

# Populate 15 days of dummy history
daily_history = {}
today = datetime.now()

models = ["mimo-v2.5-free", "kilo-auto/free", "poolside/laguna-m.1", "antigravity/gemini-3.5-flash-low"]

for i in range(15):
    date_str = (today - timedelta(days=i)).date().isoformat()
    # Skip a couple of days to test streaks
    if i in (4, 10):
        continue
        
    day_entry = {
        "sessions": 2 + (i % 3),
        "timestamps": []
    }
    
    for idx, model in enumerate(models):
        # Different models have different usage
        multiplier = (4 - idx) * 1000 * (15 - i)
        in_tok = int(120 * multiplier)
        out_tok = int(350 * multiplier)
        reqs = int(5 * (4 - idx))
        
        day_entry[model] = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "requests": reqs
        }
        
    daily_history[date_str] = day_entry

data = {
    "schema_version": 1,
    "models": {
        "mimo-v2.5-free": {
            "input_tokens": 153200000,
            "output_tokens": 353200000,
            "total_tokens": 506400000,
            "request_count": 276
        }
    },
    "daily_history": daily_history,
    "updated_at": datetime.now().isoformat() + "Z"
}

usage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Mock usage data successfully populated.")
