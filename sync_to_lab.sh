#!/bin/bash
# sync_to_lab.sh
# Pushes local context (papers, brainstorming, etc.) to the lab server's singing_identity project.
# It ensures any documents you read locally are available to the remote Codex environment.

echo "🔄 Syncing Local Context TO Lab Server (valkyrie03)..."

# Syncing non-data context and workspace folders (docs, context, design, free_recall, notebooks, tests, experiments, scripts, .agents, pro_suggestions)
rsync -avz --progress \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.vscode' \
    --exclude '__pycache__' \
    --exclude 'raw_audio' \
    --exclude '*.tar.gz' \
    --exclude '*.npz' \
    --exclude '*.wav' \
    /Users/bowen/research/project/docs \
    /Users/bowen/research/project/context \
    /Users/bowen/research/project/design \
    /Users/bowen/research/project/free_recall \
    /Users/bowen/research/project/notebooks \
    /Users/bowen/research/project/experiments \
    /Users/bowen/research/project/tests \
    /Users/bowen/research/project/scripts \
    /Users/bowen/research/project/.agents \
    /Users/bowen/research/project/pro_suggestions \
    /Users/bowen/research/project/pyproject.toml \
    /Users/bowen/research/project/uv.lock \
    valkyrie03:~/bowen_lab/projects/singing_identity/

# Syncing root markdown files (surveys, brainstorming)
rsync -avz --progress \
    /Users/bowen/research/project/*.md valkyrie03:~/bowen_lab/projects/singing_identity/

echo "✅ Sync to lab complete!"
