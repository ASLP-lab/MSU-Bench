failed_path = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/align_dirs.failed.list"
scp_result_path = "$PIPELINE_ROOT/gemini_revise/rerun/test/align_dirs.rerun.scp"

with open(failed_path, "r") as f:
    lines = f.readlines()

scps = []
seen = set()

for line in lines:
    if not line.strip():
        continue

    src, _ = line.strip().split(r"\t", 1)

    if src in seen:
        continue

    seen.add(src)
    scps.append(src)
    print(src)

with open(scp_result_path, "w") as f:
    for scp in scps:
        f.write(scp + "\n")

print(f"Written {len(scps)} unique scp entries to {scp_result_path}")