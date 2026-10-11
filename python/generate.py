#!/usr/bin/env python3
import argparse
from pathlib import Path
from sora_model import SoraRNN

def main():
    p=argparse.ArgumentParser(description="Generate from a saved Sora checkpoint.")
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--prompt",default="Sora")
    p.add_argument("--bytes",type=int,default=200)
    p.add_argument("--temperature",type=float,default=.8)
    p.add_argument("--seed",type=int,default=1)
    a=p.parse_args()
    print(SoraRNN.load(a.checkpoint).generate(a.prompt,a.bytes,a.temperature,a.seed))
if __name__=="__main__":main()
