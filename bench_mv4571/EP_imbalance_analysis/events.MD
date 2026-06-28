# Torch profiler trace (`.pt.trace.json.gz`) — schema + cách tự kiểm chứng

1 file = 1 rank: `dp<DP>_pp0_tp0_dcp0_ep<EP>_rank0.<id>.pt.trace.json.gz`.
Định dạng = **Chrome Trace Event Format** (gzip JSON). Mọi lệnh dưới dùng `ijson` để
**stream** (đừng `json.load` cả file — ~7M event → OOM).

> Đặt biến cho tiện:
> ```bash
> cd /home/phuc-nguyen/workspaces/mv-4571/auto-script/bench_mv4571/logs/glm5.2/dp8ep8/noMTP-bs64-dg/auto_profile/20260625_091838/profiling_result/run1/8k_rinf_c8
> F=$(ls dp0_*rank0*.pt.trace.json.gz | head -1)
> ```

---

## 1. Cấu trúc top-level

```jsonc
{
  "schemaVersion": 1,
  "deviceProperties": [            // 1 entry / GPU (8 GPU)
    { "id": 0, "name": "", "totalGlobalMem": 206141652992,
      "computeMajor": 9, "computeMinor": 4,   // gfx94x = MI300
      "warpSize": 64, "numSms": 304, "sharedMemPerBlock": 65536, ... },
    ...
  ],
  "traceEvents": [ {…}, {…}, … ],  // <-- TẤT CẢ event ở đây (CPU rồi GPU)
  // có thể có thêm: "distributedInfo", "traceName", "deviceProperties"...
}
```

**Check:** in các key top-level + header:
```bash
zcat "$F" | head -c 1200
# hoặc liệt kê key top-level (stream, không nạp cả file):
python3 -c "import gzip,ijson;
f=gzip.open('$F','rb');
print([k for k,_ in __import__('ijson').kvitems(f,'')][:10])" 2>/dev/null || \
python3 - <<'PY'
import gzip, ijson, os
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
keys=[]
for prefix,event,value in ijson.parse(gzip.open(F,'rb')):
    if prefix=='' and event=='map_key': keys.append(value)
    if value=='traceEvents' or len(keys)>8: break
print("top-level keys:", keys)
PY
```

---

## 2. Một event (phần tử của `traceEvents`)

Mọi event là 1 dict. **Field chung:**

| Field | Kiểu | Ý nghĩa |
|---|---|---|
| `ph` | str(1 ký tự) | **phase type** (loại event theo Chrome Trace Format) — xem §2b. Ta chỉ dùng `"X"`. |
| `cat` | str | category (xem §3) |
| `name` | str | tên op/kernel (dùng để phân loại) |
| `pid` | int/str | process id. Kineto **đặt lại**: GPU event → `pid` ≈ device id; CPU event → pid thật của process. |
| `tid` | int/str | thread id. GPU event → `tid` = **stream** (compute vs comm có thể khác tid); CPU → thread thật. |
| `ts` | float | thời điểm **bắt đầu**, đơn vị **µs**, clock **chung giữa các rank** |
| `dur` | float | thời lượng, **µs** (chỉ có ở event `ph=="X"`) |
| `args` | dict | metadata phụ (khác nhau theo `cat`, xem §3) |

### 2b. `ph` — các giá trị (Chrome Trace Event Format)

`ph` (phase) cho biết event thuộc loại nào. Trace này (đo thực) chỉ chứa **5** loại:

| `ph` | Tên | Có `dur`? | Trong trace này (cat / vd) |
|---|---|---|---|
| **`X`** | **Complete** (1 slice có bắt đầu + thời lượng) | ✅ | **cái ta dùng**: `kernel`, `cpu_op`, `cuda_runtime`, `gpu_memcpy`, `user_annotation` |
| `s` | Flow **start** (đầu mũi tên liên kết) | ❌ | `cat="ac2g"` — nối CPU-launch → GPU-kernel (theo `correlation` id) |
| `f` | Flow **finish/step** (cuối mũi tên) | ❌ | `cat="ac2g"` (rất nhiều: ~3.3M) |
| `M` | **Metadata** (đặt tên process/thread) | ❌ | `name="process_name"`, `"thread_name"` (vài chục cái) |
| `i` | **Instant** (mốc tức thời, 0 thời lượng) | ❌ | `name="Iteration Start: PyTorch Profiler"` |

Các `ph` khác của chuẩn Chrome (KHÔNG xuất hiện ở đây, để biết thêm): `B`/`E` = begin/end
cặp (kineto dùng `X` thay thế), `b`/`n`/`e` = async, `C` = counter, `P` = sample,
`N`/`O`/`D` = object, `R` = mark.

> **Vì sao chỉ lấy `ph=="X"`:** chỉ `X` mới là 1 khoảng thời gian thật (có `ts`+`dur`) để đo.
> `s`/`f` chỉ là *mũi tên liên kết* (CPU gọi → kernel GPU nào, qua `correlation`), `M`/`i`
> là metadata/mốc — không có thời lượng.

**Check phân bố `ph` (và `ph`×`cat`) toàn file:**
```bash
python3 - <<'PY'
import gzip, ijson, os, collections
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
ph=collections.Counter(); phcat=collections.Counter()
for e in ijson.items(gzip.open(F,'rb'),'traceEvents.item'):
    ph[e.get('ph')]+=1; phcat[(e.get('ph'),e.get('cat'))]+=1
print("ph:", dict(ph))            # {'X':~3.78M, 'f':~3.30M, 's':~61798, 'M':64, 'i':2}
print("ph x cat:", dict(phcat))
PY
```

---

## 3. Các giá trị `cat` (và cái nào dùng)

| `cat` | Là gì | Chạy ở | `ph` | Ta dùng? |
|---|---|---|---|---|
| `kernel` | **GPU kernel** (gemm, sorting, quant, nccl…) | GPU | X | ✅ **CHÍNH** |
| `cpu_op` | aten op phía CPU (`aten::select`…) | CPU | X | ❌ |
| `cuda_runtime` | lời gọi HIP/CUDA runtime (`hipLaunchKernel`…) | CPU | X | ❌ |
| `user_annotation` / `gpu_user_annotation` | annotation (`gloo:all_reduce`, nvtx…) | CPU/GPU | X | ❌ |
| `gpu_memcpy` / `gpu_memset` | copy/memset trên GPU | GPU | X | (không cần) |
| `ac2g` | **flow** nối CPU-launch ↔ GPU-kernel (qua `correlation`) | — | s/f | ❌ (chỉ là mũi tên) |
| (None) | metadata / instant | — | M/i | ❌ |

⚠️ **Thứ tự trong file: CPU event (`cpu_op`,`user_annotation`) đứng TRƯỚC, GPU `kernel`
nằm SAU** — nên nếu chỉ đọc prefix sẽ không thấy kernel. Phải lọc `cat=="kernel"`.

**`args` của GPU `kernel`:**
`{ "External id", "device", "stream", "correlation", "kind"="Dispatch Kernel",
   "grid"=[x,y,z], "block"=[x,y,z] }`
**`args` của `cpu_op`/`user_annotation`:** `{ "External id", "Record function id", "Ev Idx" }`.

**Check** (phân bố `cat`, `ph`, args-keys, + 1 sample mỗi cat — quét toàn file, ~1-2 phút):
```bash
python3 - <<'PY'
import gzip, ijson, os, json, collections
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
cats=collections.Counter(); phs=collections.Counter(); sample={}
for e in ijson.items(gzip.open(F,'rb'),'traceEvents.item'):
    c=e.get('cat'); cats[c]+=1; phs[e.get('ph')]+=1
    if c not in sample: sample[c]=e
print("cat:", dict(cats)); print("ph:", dict(phs))
for c,e in sample.items():
    print(c, "->", json.dumps({k:e.get(k) for k in ('ph','cat','name','pid','tid','ts','dur','args')}, default=str)[:300])
PY
```

---

## 4. Phân loại kernel MoE bằng `name` (substring)

| Loại | substring trong `name` |
|---|---|
| comm (gather/reduce-scatter) | `ncclDevKernel` |
| MoE sorting | `MoeSorting` |
| quant | `dynamic_per_group_scaled_quant` |
| gemm stage1 | `kernel_moe_gemm` **và** `(ck::InMemoryDataOperationEnum)0` |
| gemm stage2 | `kernel_moe_gemm` **và** `(ck::InMemoryDataOperationEnum)1` |
| fused MoE (decode) | `fmoe` (vd `aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_...`) |

**Check** (đếm từng loại GPU-kernel toàn file):
```bash
python3 - <<'PY'
import gzip, ijson, os, collections
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
c=collections.Counter()
for e in ijson.items(gzip.open(F,'rb'),'traceEvents.item'):
    if e.get('cat')!='kernel': continue
    n=e.get('name','')
    if 'ncclDevKernel' in n: c['comm']+=1
    elif 'MoeSorting' in n: c['sort']+=1
    elif 'dynamic_per_group_scaled_quant' in n: c['quant']+=1
    elif 'kernel_moe_gemm' in n: c['gemm1' if '(ck::InMemoryDataOperationEnum)0' in n else 'gemm2']+=1
    elif 'fmoe' in n: c['fmoe']+=1
print(dict(c))
# kỳ vọng (case 8k/c8): comm=154650, sort=78000, quant=231750, gemm1=gemm2=77100, fmoe=225
PY
```

---

## 5. Các khẳng định & cách CHECK

### (a) `ts`/`dur` đơn vị µs
1 gemm ~ vài chục µs, 1 step decode ~ vài trăm µs. In dur của gemm:
```bash
python3 - <<'PY'
import gzip, ijson, os
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
for e in ijson.items(gzip.open(F,'rb'),'traceEvents.item'):
    if e.get('cat')=='kernel' and 'kernel_moe_gemm' in e.get('name',''):
        print("gemm dur =", float(e['dur']), "us"); break   # ~30us
PY
```

### (b) Clock CHUNG giữa các rank (so timestamp giữa rank hợp lệ)
So khoảng `[ts_min, ts_max]` của 2 rank — phải overlap gần như hoàn toàn, lệch < vài ms:
```bash
python3 - <<'PY'
import gzip, ijson, glob
def rng(f):
    lo=float('inf'); hi=-1
    for e in ijson.items(gzip.open(f,'rb'),'traceEvents.item'):
        if e.get('cat')!='kernel': continue
        t=float(e['ts']); lo=min(lo,t); hi=max(hi,t)
    return lo,hi
f0=sorted(glob.glob('dp0_*rank0*.gz'))[0]; f1=sorted(glob.glob('dp1_*rank0*.gz'))[0]
a=rng(f0); b=rng(f1)
print("rank0",a); print("rank1",b); print("offset min (ms)=", (b[0]-a[0])/1e3)
PY
```

### (c) Mọi comm GIỐNG HỆT (không phân biệt gather vs reduce-scatter bằng tên/args)
```bash
python3 - <<'PY'
import gzip, ijson, os, collections
F=os.popen("ls dp0_*rank0*.pt.trace.json.gz").read().split()[0]
names=collections.Counter(); grids=collections.Counter()
for e in ijson.items(gzip.open(F,'rb'),'traceEvents.item'):
    if e.get('cat')=='kernel' and 'ncclDevKernel' in e.get('name',''):
        names[e['name']]+=1; grids[str(e.get('args',{}).get('grid'))]+=1
print("comm name variants:", dict(names))   # chỉ 1 TÊN -> không phân biệt gather/rs bằng tên
print("comm grid variants:", dict(grids))   # {'None':154200, '[112,1,1]':450} -> grid cũng KHÔNG
                                             # tách được gather vs reduce-scatter (cả hai đều grid=None
                                             # ở steady-state; 450 cái [112,1,1] là collective startup)
PY
```
→ kết luận: phải phân biệt gather vs reduce-scatter bằng **vị trí** (comm trước cụm MoE = gather,
comm sau = reduce-scatter), không dựa được vào `name`/`args`.

### (d) MAPPING: #comm bằng nhau giữa 8 rank
```bash
for f in dp*_rank0*.pt.trace.json.gz; do
  python3 - "$f" <<'PY'
import gzip, ijson, sys
n=sum(1 for e in ijson.items(gzip.open(sys.argv[1],'rb'),'traceEvents.item')
      if e.get('cat')=='kernel' and 'ncclDevKernel' in e.get('name',''))
print(sys.argv[1].split('_')[0], "#comm=", n)
PY
done   # cả 8 rank phải in cùng 1 số (154650)
```

### (e) `154650 = 2 × (77100 + 225)` → 77,325 cụm MoE
Mỗi cụm MoE có **2 comm** (1 gather + 1 reduce-scatter) ⇒ `comm = 2×#cụm`. Đếm #cụm theo
loại: cụm 2-stage có 1 gemm1 + 1 gemm2, cụm fused có 1 fmoe. Vậy:
`comm == gemm1 + gemm2 + 2×fmoe` (tổng quát, không giả định gemm1=gemm2).
Số liệu: `77100 + 77100 + 2×225 = 154650 = comm`. ✔ (#cụm = 77,325.)

### (f) SYNC ở điểm kết thúc collective (gather.end & rs.end ~bằng nhau giữa rank)
Đây là phần `analyze_trace_pairs.py` đo: spread `gather.END`/`rscat.END` giữa 8 rank ~1µs
trong khi MoE-time ~95µs. Chạy script đó (xem `TRACE_TIME_ANALYSIS.md`) → mục "Chứng minh sync".
