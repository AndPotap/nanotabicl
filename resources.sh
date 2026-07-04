echo "=== CPU Information ==="

lscpu | grep -E \
'Architecture:|Model name:|Socket\(s\):|Core\(s\) per socket:|Thread\(s\) per core:|CPU max MHz:|CPU min MHz:|cache'

echo
echo "=== SIMD Extensions ==="

flags=$(lscpu | sed -n 's/^Flags:[[:space:]]*//p')

for ext in sse sse2 sse3 ssse3 sse4_1 sse4_2 avx avx2 avx512f fma; do
    if echo "$flags" | grep -qw "$ext"; then
        echo "$ext: yes"
    else
        echo "$ext: no"
    fi
done
