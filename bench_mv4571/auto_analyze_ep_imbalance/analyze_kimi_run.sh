#!/usr/bin/env bash
# Analyze (HOST) các case ĐÃ THU XONG trong 1 run dir (sweep chạy ANALYZE=0 -> analyze tách ở host).
# Idempotent: bỏ qua case đã có summary; chỉ chạy case có đủ data. Re-run nhiều lần khi sweep tiến triển.
#   bash analyze_kimi_run.sh logs/run_20260630_095937
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${1:?usage: analyze_kimi_run.sh <run_dir>}"
LOGD="$HERE/claude-logs/artifacts"; mkdir -p "$LOGD"
conc_of(){ basename "$1" | sed -n 's/.*_c\([0-9][0-9]*\).*/\1/p'; }

echo "=== TOKEN (serve.log có [EP_COLLECT]) ==="
while IFS= read -r sl; do
  base="$(dirname "$sl")"; tag="$(echo "$base" | sed 's#.*/kimi2.6/##;s#/#_#g')"
  [ -f "$base/analysis/summary_tokens.json" ] && { echo "  skip(done) $tag"; continue; }
  ec=$(grep -c EP_COLLECT "$sl" 2>/dev/null | tr -d '\n'); ec=${ec:-0}
  [ "$ec" -lt 100 ] && { echo "  SKIP(low EP_COLLECT=$ec) $tag"; continue; }
  comp=$(python3 -c "import json,glob;f=glob.glob('$base/bench/*.json');print(json.load(open(f[0])).get('completed',0) if f else 0)" 2>/dev/null || echo 0)
  [ "${comp:-0}" -lt 1 ] && { echo "  SKIP(bench completed=$comp -> crashed/incomplete) $tag"; continue; }
  echo "  analyze token: $tag (EP_COLLECT=$ec)"
  python3 -u "$HERE/analyze_tokens.py" --log "$sl" --out "$base/analysis" \
     --decode-max-tokens auto --concurrency "$(conc_of "$(dirname "$base")")" \
     > "$LOGD/an_tok_${tag}.log" 2>&1 && echo "    OK" || echo "    FAIL (xem $LOGD/an_tok_${tag}.log)"
done < <(find "$RUN" -path "*tokens/serve.log" | sort)

echo "=== TIME (>=8 traces) ==="
while IFS= read -r td; do
  base="$(dirname "$td")"; tag="$(echo "$base" | sed 's#.*/kimi2.6/##;s#/#_#g')"
  [ -f "$base/analysis/summary_time.json" ] && { echo "  skip(done) $tag"; continue; }
  n=$(ls "$td"/*.gz 2>/dev/null | wc -l)
  [ "$n" -lt 8 ] && { echo "  SKIP(traces=$n) $tag"; continue; }
  echo "  analyze time: $tag ($n traces)"
  python3 -u "$HERE/analyze_time.py" --trace-dir "$td" --out "$base/analysis" \
     --phase-method cudagraph --no-verify-schema --no-gantt \
     > "$LOGD/an_time_${tag}.log" 2>&1 && echo "    OK" || echo "    FAIL (xem $LOGD/an_time_${tag}.log)"
done < <(find "$RUN" -path "*time/traces" -type d | sort)
echo "DONE analyze_kimi_run @ $(date '+%T')"
