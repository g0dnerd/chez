#!/usr/bin/env fish
#
# merge_datasets.fish -- concatenate selfplay .bin datasets into one, validate
# the result, and (only on success) delete the input parts.
#
# Records are fixed 35-byte (see selfplay.zig), so concatenating valid datasets
# yields a valid dataset. Each part is validated *before* merging so a truncated
# part can't silently misalign records across a boundary (two parts each off by
# a few bytes could otherwise sum to a multiple of 35 and pass a final check
# while being corrupt at the seam). The merged file is validated again before
# any part is removed, so a failure anywhere leaves all inputs untouched.
#
# Usage:
#   scripts/merge_datasets.fish <merged.bin> <part1.bin> [part2.bin ...]

set -l script_dir (dirname (status filename))
set -l validate $script_dir/validate_dataset.fish

if test (count $argv) -lt 2
    echo "usage: $argv0 <merged.bin> <part1.bin> [part2.bin ...]" >&2
    exit 1
end

set -l out $argv[1]
set -l parts $argv[2..-1]

if test -e $out
    echo "error: output already exists: $out (refusing to overwrite)" >&2
    exit 1
end

# Validate every part up front, and refuse if a part is also the output target.
for p in $parts
    if test $p = $out
        echo "error: part path equals output path: $p" >&2
        exit 1
    end
    if not fish $validate $p
        echo "error: part failed validation, nothing merged: $p" >&2
        exit 1
    end
end

# Merge into a temp file beside the output, validate, then publish atomically.
set -l tmp "$out.partial"
rm -f $tmp
if not cat $parts >$tmp
    echo "error: merge (cat) failed" >&2
    rm -f $tmp
    exit 1
end

if not fish $validate $tmp
    echo "error: merged dataset failed validation; parts left untouched" >&2
    rm -f $tmp
    exit 1
end

mv $tmp $out

# Validation passed -- safe to remove the parts now.
for p in $parts
    rm -f $p
end

echo "merged "(count $parts)" part(s) into $out; parts deleted"
fish $validate $out
