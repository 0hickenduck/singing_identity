# Repo migration checklist — arti6_linearvc → singing_identity

**Date:** 2026-07-16
**Decisions locked:** new repo name `singing_identity`; **private**; `docs/pdf/` never enters
git (rsync only); old GitHub repo `0hickenduck/arti6_linearvc` stays as-is (not archived, not
deleted).
**Why fresh history:** the current local `.git` is **8.8 GB** (old `external/`, `.trellis/`,
`archive/`, `vtuber_pipeline/`, demo trees baked into history). GitHub rejects >100 MB blobs and
the dead weight has no research value — provenance lives in `context/EXPERIMENT_REGISTRY.md`
and `results/`. The old history is archived locally, not rewritten.

---

## 0. Where and when to run

- Run on the **Mac** at `/Users/bowen/research/project`. Do **not** run from the Windows
  Google-Drive mirror (`G:\其他计算机\My Mac\project`).
- **Pause Google Drive sync** (menu-bar Drive icon → 暂停同步) before step 2, resume after
  step 4. A `.git` swap racing the sync client is the one way this can corrupt.
- `.gitignore` was already rewritten on 2026-07-16 (adds `lab/`, `output/`, `tmp/`,
  `docs/pdf/`, `*.npz/*.npy/*.pt/*.pth/*.wav/*.tar.gz`, `.DS_Store`, `*.log`). It reaches the
  Mac via Drive — verify it's there before starting: `grep docs/pdf .gitignore`.

## 1. (Recommended) push the old repo's tail, so its final state also exists remotely

```bash
cd /Users/bowen/research/project
git fetch origin
git log --oneline origin/master..HEAD        # any unpushed commits?
git push origin master                        # recent commits are text-only; should be quick
```

If the push is rejected (oversized historical blob), skip it — step 2 preserves everything
locally anyway.

## 2. Move the old history OUT of the Drive-synced folder

```bash
mkdir -p ~/research/_git_archives
mv .git ~/research/_git_archives/arti6_linearvc.git-20260716
```

Outside the synced tree: Drive stops re-uploading 8.8 GB, and no `.gitignore` entry is needed.
**Rollback at any point before step 4 completes:**
`mv ~/research/_git_archives/arti6_linearvc.git-20260716 .git`

## 3. Fresh init and audited first commit

```bash
git init -b main
git add .

# AUDIT before committing — all three must look right:
git check-ignore -v docs/pdf/ lab/ output/ tmp/ | cat      # all four listed as ignored
git status --short | head -50                               # no docs/pdf, no lab/, no *.npz
find . -type f -size +20M -not -path "./.git/*" -exec git ls-files --error-unmatch {} \; 2>/dev/null # expect: empty output

git commit -m "singing_identity: initial import (paper closure 2026-07-15 + supplement plan 2026-07-16)"
du -sh .git                                                  # expect tens of MB, not GB
```

If `find` lists an unexpected big file, add an ignore rule, `git rm --cached <file>`, re-audit,
then commit.

## 4. Create the private GitHub repo and push

With `gh` (one command creates repo + remote + pushes):

```bash
gh auth status || gh auth login
gh repo create 0hickenduck/singing_identity --private --source=. --remote=origin --push
```

Without `gh`: create an **empty** private repo `singing_identity` on github.com (no README, no
license — the tree already has both), then:

```bash
git remote add origin git@github.com:0hickenduck/singing_identity.git
git push -u origin main
```

Then **resume Google Drive sync**.

## 5. Lab server adopts the new repo (valkyrie03)

The lab tree at `~/bowen_lab/projects/singing_identity` currently has its own unrelated git
history (execution commits such as `7839dee`). Archive it and adopt the GitHub history without
touching the working tree:

```bash
ssh valkyrie03
ssh -T git@github.com                        # confirm a GitHub key exists; add one if not
cd ~/bowen_lab/projects/singing_identity
mkdir -p ~/_git_archives
mv .git ~/_git_archives/singing_identity_lab.git-20260716

git clone --no-checkout git@github.com:0hickenduck/singing_identity.git /tmp/si_fresh
mv /tmp/si_fresh/.git .git && rm -rf /tmp/si_fresh
git reset                                    # index = GitHub main; working tree = lab tree, untouched

# Files that exist on main but are MISSING from the lab tree show as "D" — materialize them
# (this is how today's supplement plan & survey reach the lab if rsync hasn't run):
git status --porcelain | awk '$1=="D"{print $2}' | xargs -r git checkout --

git diff --stat                              # remaining diffs = real content drift; review, don't blind-checkout
git status --short | head -40                # untracked leftovers = lab-only artifacts (full results dirs etc.)
```

Lab-only deltas (e.g. complete `results/*/` contents that the laptop excludes) are governed by
the same `.gitignore`; commit the small ones worth versioning, leave the rest. From now on the
lab commits/pushes to `origin main` like any clone.

## 6. Post-migration bookkeeping

- [ ] `README.md`: title/description say singing_identity, note the remote and the
      git-vs-rsync split.
- [ ] `AGENTS.md` + `LAB_SERVER_WORKFLOW.md`: one line each — *code/docs move via git
      (`github.com/0hickenduck/singing_identity`, private); features, score matrices, and
      `docs/pdf/` move via `sync_to_lab.sh`/`sync_from_lab.sh` only.*
- [ ] Day-to-day from here: **git** for code/context/results-compact (Mac ↔ GitHub ↔ lab);
      **rsync** unchanged for heavy data and PDFs; the Windows Drive mirror stays read-only.
- [ ] Old repo `0hickenduck/arti6_linearvc`: left exactly as-is per decision.

---

**One-glance risk table**

| Risk | Guard |
|---|---|
| Drive sync corrupts `.git` mid-swap | Pause sync (step 0); swap is two `mv`s |
| Oversized blob sneaks into first commit | step 3 `find -size +20M` audit before commit |
| Copyrighted PDFs pushed | `docs/pdf/` ignored; audit in step 3 double-checks |
| Lab execution history lost | archived to `~/_git_archives/...-20260716`, never deleted |
| Anything goes wrong pre-push | rollback one-liner in step 2 |
