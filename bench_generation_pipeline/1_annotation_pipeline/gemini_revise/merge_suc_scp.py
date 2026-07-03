scp_files = [
    "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/align_dirs.success.list",
    "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/revised/align_dirs.success.list",
    "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/rerun/align_dirs.success.list",
]

out_scp = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/all_sucess.scp"

seen = set()
merged = []

def read_scp(path):
    with open(path, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                yield p

for scp in scp_files:
    for p in read_scp(scp):
        if p not in seen:
            seen.add(p)
            merged.append(p)

with open(out_scp, "w") as f:
    for p in merged:
        f.write(p + "\n")

print(f"Merged {len(scp_files)} scp files → {out_scp}")
print(f"Total unique paths: {len(merged)}")