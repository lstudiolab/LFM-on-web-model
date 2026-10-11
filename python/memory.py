"""Simple explicit persistent memory store for Sora.

This is external symbolic memory, not neural weights. Learned model knowledge
is stored in model checkpoints; user-specific remembered facts are stored here
only when explicitly added by the application/user.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

class MemoryStore:
    def __init__(self, path="sora-memory.json"):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"format": "lfm-sora-memory-v1", "memories": []}

    def add(self, text, category="note", source="user"):
        text = text.strip()
        if not text: raise ValueError("Memory text cannot be empty")
        item = {"text": text, "category": category, "source": source,
                "created_at": datetime.now(timezone.utc).isoformat()}
        self.data["memories"].append(item)
        self.save()
        return item

    def search(self, query, limit=10):
        terms = set(query.casefold().split())
        scored = []
        for item in self.data["memories"]:
            words = set(item["text"].casefold().split())
            score = len(terms & words)
            if score: scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    def forget(self, index):
        del self.data["memories"][index]
        self.save()
