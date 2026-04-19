#!/usr/bin/env bash
# One-time setup: venv, deps, git, GitHub repo, Pages.
set -euo pipefail

cd "$(dirname "$0")"

REPO_NAME="explain-market-today"

echo "==> Python venv + deps"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> Ollama model check"
if ! ollama list | grep -q "qwen2.5:14b"; then
  echo "qwen2.5:14b not found. Pulling…"
  ollama pull qwen2.5:14b
fi

echo "==> Prep docs/ (GitHub Pages root)"
mkdir -p docs/episodes
touch docs/.nojekyll

echo "==> Git init"
if [ ! -d .git ]; then
  git init -b main
  cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.state.json
.DS_Store
EOF
  git add .
  git commit -m "initial: market podcast generator"
fi

echo "==> GitHub repo (public, required for free Pages)"
if ! gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --public --source=. --push
else
  echo "Repo exists. Ensuring remote + push."
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/$(gh api user -q .login)/$REPO_NAME.git"
  git push -u origin main || true
fi

echo "==> Enable GitHub Pages on main /docs"
gh api -X POST "repos/$(gh api user -q .login)/$REPO_NAME/pages" \
  -f "source[branch]=main" -f "source[path]=/docs" 2>/dev/null || \
  gh api -X PUT "repos/$(gh api user -q .login)/$REPO_NAME/pages" \
    -f "source[branch]=main" -f "source[path]=/docs" 2>/dev/null || \
  echo "Pages may already be enabled. Check repo settings."

USER="$(gh api user -q .login)"
URL="https://$USER.github.io/$REPO_NAME"
echo ""
echo "==> Done."
echo "Pages URL: $URL"
echo "Feed URL:  $URL/feed.xml"
echo ""
echo "If PODCAST_BASE_URL in config.py differs from above, edit it to match."
echo ""
echo "Submit feed to platforms:"
echo "  Spotify:  https://podcasters.spotify.com/submit"
echo "  Apple:    https://podcastsconnect.apple.com"
