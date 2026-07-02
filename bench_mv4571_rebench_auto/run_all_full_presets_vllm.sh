# EPLB sweep (async only): communicator {pynccl,nccl,nixl} x step {default,s250} x
# num_redundant {r0,r8,r16}, workload fixed by env.yaml ([64]x[100k]). A crashing
# preset (e.g. nixl) aborts fast via SERVER_FATAL_RE and the next line runs.
# Regenerate presets with: bash gen_eplb_presets.sh

# --- Priority: baseline + 3 communicators (default step, no redundancy) ---
PRESET=glm5.2.rebench/MTP5-bs64-dg.yaml bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-default-r0.yaml bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-default-r0.yaml   bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-default-r0.yaml   bash run_all.sh

# # --- Redundancy sweep (default step) ---
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-default-r8.yaml  bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-default-r16.yaml bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-default-r8.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-default-r16.yaml   bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-default-r8.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-default-r16.yaml   bash run_all.sh

# # --- Step-250 sweep (window_size/step_interval = 250) ---
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-s250-r0.yaml  bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-s250-r8.yaml  bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-pynccl-async-s250-r16.yaml bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-s250-r0.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-s250-r8.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nccl-async-s250-r16.yaml   bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-s250-r0.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-s250-r8.yaml    bash run_all.sh
# PRESET=glm5.2.rebench/MTP5-bs64-dg-eplb-nixl-async-s250-r16.yaml   bash run_all.sh
