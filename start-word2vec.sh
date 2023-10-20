#!/bin/bash
cd /root/word2vec-api && python3 ./word2vec-api.py --model ./models/newfile.txt  --port 8082  2>&1 > /root/word2vec-logs.txt & 
