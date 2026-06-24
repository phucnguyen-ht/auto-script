################# ENABLE CUDA GRAPH
# ### DP8EP8 -- GLM-5.2 EP-balancing sweep (MV-4571): MTP 0 and MTP 3.
PRESET=glm5.2/dp8ep8/noMTP-bs64-dg.yaml bash run_all.sh
PRESET=glm5.2/dp8ep8/MTP5-bs64-dg.yaml  bash run_all.sh
