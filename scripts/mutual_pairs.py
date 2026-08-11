"""Ported from $vaultarea/security/Engagements/memsom-core-baseline-2026-07-31/evidence/edges.py
(PLAN.md Phase 0, Section 9.5) -- repointed at the local checkout instead of a
hardcoded ~/memsom, everything else unchanged."""
import ast, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "memsom"
PKGS = {"interface","bridge","federation","distill","lifecycle","retrieval","integrity","storage"}

def modname(p):
    rel = p.relative_to(ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__": parts.pop()
    return ".".join(parts)

def pkg(mod):
    parts = mod.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in PKGS else "__core__"

edges = collections.Counter(); detail = collections.defaultdict(list)
for p in ROOT.rglob("*.py"):
    if "__pycache__" in str(p): continue
    me = modname(p); src = pkg(me)
    try: tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError: continue
    for n in ast.walk(tree):
        targets = []
        if isinstance(n, ast.ImportFrom):
            if n.level:                                  # relative -> resolve
                base = me.split(".")
                if p.name != "__init__.py": base = base[:-1]
                base = base[:len(base)-(n.level-1)] if n.level > 1 else base
                full = ".".join(base + ([n.module] if n.module else []))
            else:
                full = n.module or ""
            if full.startswith("memsom"):
                targets.append(full)
                for a in n.names: targets.append(f"{full}.{a.name}")
        elif isinstance(n, ast.Import):
            targets += [a.name for a in n.names if a.name.startswith("memsom")]
        for t in targets[:1] if targets else []:
            d = pkg(t)
            if d != src:
                edges[(src,d)] += 1
                detail[(src,d)].append(f"{p.relative_to(ROOT)}:{n.lineno} -> {t}")

print("=== fan-in on the frozen core (__core__) ===")
tot = collections.Counter()
for (s,d),c in edges.items():
    if d=="__core__": tot[s]+=c
for s,c in tot.most_common(): print(f"  {s:12} {c}")
print("  TOTAL", sum(tot.values()))

print("\n=== mutual pairs (subpackage <-> subpackage) ===")
seen=set()
pair_count = 0
for (s,d),c in sorted(edges.items()):
    if "__core__" in (s,d): continue
    if (d,s) in edges and (d,s) not in seen and (s,d) not in seen:
        seen.add((s,d)); seen.add((d,s))
        pair_count += 1
        print(f"  {s:11} <-> {d:11} ({c} / {edges[(d,s)]})")
print("  TOTAL PAIRS", pair_count)
