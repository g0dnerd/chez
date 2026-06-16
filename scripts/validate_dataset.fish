#!/usr/bin/env fish
#
# validate_dataset.fish -- sanity-check a selfplay .bin dataset and report
# the number of positions it contains.
#
# A selfplay record is 35 bytes (32 pos + 2 score + 1 wdl, must match
# selfplay.zig). A sane file's size is therefore an exact multiple of 35.
#
# Usage:
#   scripts/validate_dataset.fish <dataset.bin>

set -l record_size 35

if test (count $argv) -ne 1
    echo "usage: $argv0 <dataset.bin>" >&2
    exit 1
end

set -l file $argv[1]

if not test -f $file
    echo "error: no such file: $file" >&2
    exit 1
end

set -l bytes (stat -c %s $file)

if test $bytes -eq 0
    echo "error: $file is empty" >&2
    exit 1
end

set -l remainder (math "$bytes % $record_size")

if test $remainder -ne 0
    echo "error: $file size ($bytes bytes) is not a multiple of $record_size -- corrupt or truncated dataset" >&2
    exit 1
end

set -l positions (math "$bytes / $record_size")
echo "$file: ok, $positions positions ($bytes bytes)"
