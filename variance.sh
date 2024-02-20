#!/usr/bin/env sh
cut -d ':' -f2 | awk '{$1=$1}$1' | awk -v s1=$1 '{s0=$0;s+=(s1-s0)^2}END{print(sqrt(s/NR))}'