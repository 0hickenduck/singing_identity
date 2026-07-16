#!/bin/bash
# sync_from_lab.sh
# Pulls the exact directory structure from valkyrie03's singing_identity project to the local project.
# Excludes heavy features (.npz, .pt, .wav) and environment/git state to avoid overriding local branches or downloading gigabytes of data.

echo "🔄 Syncing FROM Lab Server (valkyrie03) to Local..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.vscode' \
    --exclude '__pycache__' \
    --exclude 'lab' \
    --exclude 'docs' \
    --exclude 'demos' \
    --exclude 'results/*/' \
    --exclude 'experiments/*/results/' \
    --exclude 'experiments/*/manifests/' \
    --exclude '*.log' \
    --exclude '*.npz' \
    --exclude '*.npy' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.wav' \
    valkyrie03:~/bowen_lab/projects/singing_identity/ /Users/bowen/research/project/

echo "✅ Sync from lab complete!"
