# memsom demo crib — explain/revoke vertical slice (demo locked 2026-06-12)
# Run line by line in Windows Terminal. Reseed fresh, rehearse once, then DON'T re-seed.
#
# PRE-TAKE CHECKLIST:
#   - close anything holding memdag.db (DB browser, stale python shell)
#   - id-drift guard: the answer id is whatever "stored as node [N]" just printed —
#     if you fat-finger an extra ask, explain THAT N, not the crib's number
#   - mid-take screwup => `python memsom.py seed --reset --offline` and restart the
#     take (the DON'T-re-seed rule is about pre-take rehearsal only)
$env:PYTHONUTF8 = '1'
Set-Location $PSScriptRoot

# 0) fresh store (off camera for the 30s cut; on camera for the long cut)
# --offline = the committed snapshot of the real README: byte-identical to every
# rehearsal, zero network on camera. The tested path IS the recorded path.
python memsom.py seed --reset --offline
# beat: three memories, three channels — trust was stamped by HOW each memory
#       arrived; nobody judged the content.

python memsom.py dump
# beat: 3 nodes, 3 channels, ZERO edges — no derivation has happened yet.

# 1) ask
python memsom.py ask "How should I configure Nebula?"
# beat: the answer is only as trustworthy as its sketchiest ingredient — integrity: EXTERNAL.

# 2) explain
python memsom.py explain 4
# beat: every answer knows where it came from — sources, channels, dates; [3] sets the floor.

# 3) revoke — dry run first, then apply
python memsom.py revoke 3 --reason "untrusted source retracted"
# beat: the DAG computes the blast radius BEFORE anything dies.
python memsom.py revoke 3 --reason "untrusted source retracted" --yes
# beat: kill the source, and everything derived from it dies with it.

# 4) re-ask — THE MONEY SHOT
python memsom.py ask "How should I configure Nebula?"
# beat: external claims GONE and integrity ROSE to USER — revoking the sketchy
#       source made the answer MORE trustworthy. min(3,2) instead of min(3,2,0).

python memsom.py explain 5
# beat: fresh derivation, honest floor.

# 5) closer
python memsom.py explain 4
# beat: revocation is not amnesia — history and blame survive. The closest OSS
#       relative can forget a memory; it structurally cannot unremember the
#       conclusions built on it — its edges are merge-history, not derivation.

# Prepared answer for "the revoked text is still readable — is that deletion?":
#   Tombstone = removed-from-use + audit trail. Payload-destroying REDACTION is
#   the separate Guarantee-#6 mode (destroys content, preserves DAG shape so
#   blame still works). Deliberately out of this slice.
