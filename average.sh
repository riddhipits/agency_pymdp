#!/usr/bin/env sh
cut -d ':' -f2 | awk '{$1=$1}$1' | awk '{s+=$0}END{print s/NR}'
