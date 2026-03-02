#!/bin/bash
set -e

SRC_ROOT="/media/li/data21/wqh/limit_switch"
DST_ROOT="/media/li/data21/wqh/limit_switch_split"

# 1. 建目录
mkdir -p "$DST_ROOT"/train/GOOD/{rgb,xyz}
mkdir -p "$DST_ROOT"/test/GOOD/{rgb,xyz}
for defect in deformation impact_damage pit scratch; do
  mkdir -p "$DST_ROOT"/test/"$defect"/{rgb,xyz,gt}
done

# 2. 拆分 OK（正常）样本
ok_dirs=( "$SRC_ROOT"/OK/S* )
n_ok=${#ok_dirs[@]}
n_train=$(( n_ok * 80 / 100 ))

for i in "${!ok_dirs[@]}"; do
  dir=${ok_dirs[$i]}
  if (( i < n_train )); then
    dst="$DST_ROOT/train/GOOD"
  else
    dst="$DST_ROOT/test/GOOD"
  fi
  # copy 所有 *RGB*.jpg
  cp "$dir"/*RGB*.jpg  "$dst/rgb/"
  # copy depth tiff
  cp "$dir"/*.tiff     "$dst/xyz/"
done

# 3. 拷贝每种缺陷到 test 下，并抓取对应的 PNG 掩码
for defect in deformation impact_damage pit scratch; do
  for dir in "$SRC_ROOT"/NG/"$defect"/S*; do
    # RGB
    cp "$dir"/*RGB*.jpg  "$DST_ROOT"/test/"$defect"/rgb/
    # depth
    cp "$dir"/*.tiff     "$DST_ROOT"/test/"$defect"/xyz/
    # ground‑truth mask：匹配 NG 路径下的 *.png
    # 只复制文件名里包含 "_NG_" 的 PNG（排除其它可能的 png）
    for mask in "$dir"/*_NG_*.png; do
      [ -f "$mask" ] && cp "$mask" "$DST_ROOT"/test/"$defect"/gt/
    done
  done
done

echo "划分完成，结果在：$DST_ROOT"
