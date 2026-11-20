#!/usr/bin/env bash
# Install a verified official Recovery M0 sparse Actions source bundle
# without downloading any Git blobs on AutoDL. Default is DRY RUN.
set -euo pipefail

ROOT="/root/autodl-tmp/vla_finetuning"
ZIP="$ROOT/recovery-m0-sparse-832d54d.zip"
SOURCE_SHA="832d54dfc43f0c9e4240bb8d92f5fda7001bb093"
ZIP_SHA256="28b51a00e79ed15db3d2647bfccea34704e314fcab0adcc9c2fdd8e8a875f085"
TAR="recovery-m0-sparse-832d54d.tar.gz"
SOURCE="$ROOT/vendor/LIBERO-Recovery-$SOURCE_SHA"
echo "SOURCE_BUNDLE=$ZIP"
echo "TARGET_SOURCE=$SOURCE"
echo "PINNED_COMMIT=$SOURCE_SHA"

if [[ "$#" -ne 1 || "$1" != "--apply" ]]; then
  echo "DRY RUN: upload the exact ZIP to $ZIP, then: bash scripts/install_recovery_from_bundle.sh --apply"
  exit 0
fi
[[ "$PWD" == "$ROOT" ]] || { echo "STOP: cd $ROOT first"; exit 2; }
[[ -s "$ZIP" ]] || { echo "STOP: ZIP is absent at $ZIP. Download it from ChatGPT/GitHub Actions and upload to AutoDL/JupyterLab first."; exit 2; }
command -v unzip >/dev/null || { echo "STOP: unzip command missing"; exit 2; }
command -v git >/dev/null || { echo "STOP: git missing"; exit 2; }
ACTUAL_ZIP_SHA="$(sha256sum "$ZIP" | awk '{print $1}')"
[[ "$ACTUAL_ZIP_SHA" == "$ZIP_SHA256" ]] || { echo "STOP: ZIP SHA256 mismatch; reject bundle"; exit 2; }

# The prior AutoDL checkout can still be blocked on repeated Git blob fetches.
# Refuse to rename source while that process remains alive.
if ps -eo args | grep -Eq '^git -C /root/autodl-tmp/vla_finetuning/vendor/LIBERO-Recovery-.* checkout'; then
  echo "STOP: old checkout process is running. Press Ctrl+C in its original terminal before applying bundle."
  exit 2
fi
if [[ -e "$SOURCE" ]] && [[ "$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null || true)" == "$SOURCE_SHA" ]] && \
  [[ -f "$SOURCE/LIBERO/setup.py" ]] && \
  [[ -f "$SOURCE/LIBERO/libero/libero/assets/stable_hope_objects/ketchup/ketchup.xml" ]] && \
  [[ -f "$SOURCE/LIBERO/libero/libero/assets/stable_scanned_objects/basket/basket.xml" ]]; then
  echo "ALREADY_PINNED_SOURCE_PRESENT; skip replacement"
  exit 0
fi

mkdir -p "$ROOT/runs" "$ROOT/vendor"
WORK="$(mktemp -d "$ROOT/runs/.m0_unpack_XXXXXXXX")"
STAGE="$(mktemp -d "$ROOT/vendor/.m0_source_XXXXXXXX")"
cleanup() {
  rm -rf -- "$WORK" "$STAGE"
}
trap cleanup EXIT
unzip -q "$ZIP" -d "$WORK"
(
  cd "$WORK"
  sha256sum -c "$TAR.sha256"
)
tar --no-same-owner -xzf "$WORK/$TAR" -C "$STAGE"
[[ "$(git -C "$STAGE" rev-parse --verify HEAD)" == "$SOURCE_SHA" ]] || { echo "STOP: pinned commit mismatch"; exit 2; }
for p in \
  "LIBERO/setup.py" \
  "LIBERO/libero/libero/assets/stable_hope_objects/ketchup/ketchup.xml" \
  "LIBERO/libero/libero/assets/stable_scanned_objects/basket/basket.xml"; do
  [[ -f "$STAGE/$p" ]] || { echo "STOP: required asset missing: $p"; exit 2; }
done

if [[ -e "$SOURCE" ]]; then
  [[ -d "$SOURCE/.git" ]] || { echo "STOP: existing source path not a recognized Git directory"; exit 2; }
  BACKUP="$SOURCE.interrupted.$(date +%s).$$"
  [[ ! -e "$BACKUP" ]] || { echo "STOP: backup path collision"; exit 2; }
  mv "$SOURCE" "$BACKUP"
  echo "BACKUP_OF_INTERRUPTED_SOURCE=$BACKUP"
fi
mv "$STAGE" "$SOURCE"
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SOURCE_SHA" ]] || { echo "STOP: post-install commit mismatch"; exit 2; }
echo "PINNED_SOURCE_VERIFIED=$SOURCE_SHA"
echo "NEXT: VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_recovery_sim_env.sh --apply"
