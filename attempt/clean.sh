#!/bin/bash
path="/home/linzhisheng/ATTEMPT/attempt"


# 函数：递归删除目录
delete_checkpoint_dirs() {
  local current_dir="$1"

  # 遍历当前目录下的所有项目
  for item in "$current_dir"/*; do
    # 检查项目类型
    if [[ -d "$item" ]]; then
      # 如果目录名称包含"checkpoint"，则删除该目录
      if [[ "$item" == *checkpoint* ]]; then
        rm -rf "$item"
        echo "Deleted directory: $item"
      else
        # 递归调用函数，检查子目录
        delete_checkpoint_dirs "$item"
      fi
    fi
  done
}

delete_checkpoint_dirs $1