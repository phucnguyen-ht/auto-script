OUTDIR="scripts/ep_balance/results"
LOGDIR="scripts/ep_balance/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

NUM_JOBS=8

for INDEX in $(seq 0 $((NUM_JOBS-1))); do
    echo "Launching job $INDEX..."

    (
        HIP_VISIBLE_DEVICES=${INDEX} python scripts/ep_balance/worker.py \
            --job-path scripts/ep_balance/jobs/job_split_${INDEX}.pt \
            --output-path ${OUTDIR}/result_split_${INDEX}.pt \
            --sample-input-path scripts/ep_balance/inputs.pt \
        2>&1 | tee "${LOGDIR}/job_split_${INDEX}.log"
    ) &
done

echo "Waiting for all jobs to finish..."
wait
echo "All jobs completed!"
echo "Results are saved in ${OUTDIR}"
