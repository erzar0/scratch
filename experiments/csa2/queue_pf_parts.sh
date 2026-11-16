#!/bin/bash
# The V4.1 reference checkpoint 65614cd9 under our exact picks (b/train_csa2_ours.py)
# with the parameter-free block's parts put in (eval/pf_parts_eval.py): its learned
# sink fixed at 0 or removed (-inf); its compressor made mean pooling of its own kv;
# both. Base: nothing changed. Run from experiments/csa2:
#   setsid nohup ./queue_pf_parts.sh < /dev/null > logs/queue_pf_parts.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_parts.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
REF=65614cd9-0fdb-4adc-b897-1294ab8caf74
for job in "learned 0 parts_base" "0 0 parts_sink0" "-inf 0 parts_sinkoff" \
    "learned 1 parts_pool" "0 1 parts_pool_sink0"; do
    read -r sink pool tag <<< "$job"
    out=logs/rule_${REF:0:8}_$tag
    [ -e $out.csv ] && continue
    ( [[ $sink == learned ]] || export SINK=$sink
      POOL=$pool CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/pf_parts_eval.py \
        $R/experiments/csa2/b/train_csa2_ours.py $REF 128 $out.csv exact ) \
        > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $tag exit $status: $(tail -n 1 $out.csv)"
done
echo "$(date +%H:%M) done"
