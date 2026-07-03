import os

a_scp = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/align_dirs.list"
b_scp = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/all_sucess.scp"
out_scp = "$PIPELINE_ROOT/gemini_revise/tmp/step4_2/all_failed.scp"

def normalize(p):
    return os.path.realpath(os.path.normpath(p.rstrip("/")))

# 读取 B，构造集合
b_set = set()
with open(b_scp) as f:
    for line in f:
        p = line.strip()
        if p:
            b_set.add(normalize(p))

# 遍历 A，做差集
diff = []
with open(a_scp) as f:
    for line in f:
        raw = line.strip()
        if not raw:
            continue
        if normalize(raw) not in b_set:
            diff.append(raw)

with open(out_scp, "w") as f:
    for p in diff:
        f.write(p + "\n")

print(f"A - B = {len(diff)} entries → {out_scp}")