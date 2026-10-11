#!/usr/bin/env python3
import argparse
from memory import MemoryStore

def main():
    ap = argparse.ArgumentParser(description="Manage Sora's explicit persistent memory.")
    sub = ap.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("text")
    add.add_argument("--category", default="note")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    sub.add_parser("list")
    forget = sub.add_parser("forget")
    forget.add_argument("index", type=int, help="Zero-based memory index")
    ap.add_argument("--file", default="sora-memory.json")
    args = ap.parse_args()
    store = MemoryStore(args.file)
    if args.command == "add":
        store.add(args.text, args.category)
        print("Memory saved.")
    elif args.command == "search":
        for item in store.search(args.query, args.limit): print(f"[{item['category']}] {item['text']}")
    elif args.command == "list":
        for i, item in enumerate(store.data["memories"]): print(f"{i}: [{item['category']}] {item['text']}")
    else:
        store.forget(args.index)
        print("Memory deleted.")

if __name__ == "__main__": main()
