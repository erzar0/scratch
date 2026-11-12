#!/bin/bash
# B exact with picks once per group of 4 layers (layers 0, 4, 8 pick with their own
# q; the others reuse them): 3 pick calls per decoded token instead of 5. After
# queue_idle.sh: untrained probe (B exact run 5f0cf619 under shared picks; gate:
# all-position loss < 3.70, B exact's own 3.599), decode kernel test (routed picks
# too), decode attention by warps, the
# decode bench (new row), the full run, per-position validation. Run from
# experiments/csa2 with setsid.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
R91PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_idle.pid)" 2>/dev/null; do sleep 30; done
S=train_tournament_kv_pick4_fast.py
sed -e 's/if layer % 2 == 0:/if layer % 4 == 0:/' \
    -e 's/even layers pick with their own q, odd reuse/its first layer picks, the rest reuse/' \
    b/train_tournament_kv_fast.py > b/$S
[ "$(grep -c 'if layer % 4 == 0:' b/$S)" = 2 ] || { echo "patch failed"; exit 1; }
eval_rule() {  # run, out tag
    CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src $R91PY eval/rule_eval.py \
        $HERE/b/$S $1 128 logs/rule_${1:0:8}_$2.csv exact > logs/rule_${1:0:8}_$2.log 2>&1
}
eval_rule 5f0cf619-be55-44ea-ad17-e182a1b0858a pick4
probe=$(grep -E ",0,8192," logs/rule_5f0cf619_pick4.csv | cut -d, -f5)
echo "probe $probe"
(cd $R && uv run python -m tests.nanogpt.test_csa2_decode > $HERE/logs/test_csa2_decode.log 2>&1)
echo "decode test exit $?"
(cd $R && PYTHONPATH=src $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/attend_warps.py > $HERE/logs/attend_warps.log 2>&1)
echo "warps exit $?"
(cd $R && PYTHONPATH=src $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_csa2.py runs/nanogpt/long_context/csa2_decode.csv \
    > $HERE/logs/decode_bench.log 2>&1)
echo "decode exit $?"
awk -v p="$probe" 'BEGIN { exit !(p != "" && p < 3.70) }' || { echo "probe failed"; exit 1; }
b/run_b.sh exact $S > logs/full_pick4.log 2>&1 || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_pick4.log | cut -c6-41)
echo "run $run"
eval_rule $run exact
echo done
