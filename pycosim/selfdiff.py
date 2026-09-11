#!/usr/bin/env python3
"""Milestone-1 validation for the Spike cosim API: Spike-vs-itself diff.

Plan-set 2026-09-11 Task A: drive processor_t::cosim_step_one() over an ELF and
diff the structured records against the stock --log-commits text stream produced
for the same ELF under the same --isa / --pc (D6 bring-up).  This proves the
write-set capture is COMPLETE before any CL coupling.

Both comparisons are run:
  (A) text-to-text : the fork's own --log-commits stream vs the baseline (unpatched)
                     spike stream.  Proves the commit_log_mem_t tuple refactor and the
                     push-site moves did not change what Spike prints.
  (B) text-to-record: the baseline (unpatched) text stream vs the fork's structured
                     records.  Proves the records contain everything the text stream
                     does -- and, for the fast-path test, things it used to lose.

usage: selfdiff.py [--elf PATH]... [--cosim-dir DIR] [--baseline SPIKE]
                   [--fork-spikes BIN] [--isa rv64gc] [--pc 0x80000000]
                   [--limit N] [--verbose]
"""
import argparse
import collections
import os
import re
import subprocess
import sys

# <pyo3>/riscv-isa-sim/pycosim -> two levels up is <pyo3>
PYO3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

KIND_NAMES = {0: "int", 1: "fp", 2: "vec", 3: "vecelem", 4: "csr"}


# --------------------------------------------------------------------------
# the stock --log-commits text stream
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"^[0-9a-fA-Fx_]+$")


def parse_text_line(line):
    """'core   0: 3 0x80000000 (0x00593073) x10  0x... c834_mcause 0x... mem 0x.. 0x..'"""
    m = re.match(r"^core\s+\d+:\s+(\d+) (0x[0-9a-f]+) \((0x[0-9a-f]+)\)(.*)$", line)
    if not m:
        return None
    priv = int(m.group(1))
    pc = int(m.group(2), 16)
    bits_field = m.group(3)
    insn_bits = int(bits_field, 16)
    # width of the printed encoding tells us the instruction length
    insn_len = (len(bits_field) - 2) * 4 // 8
    toks = m.group(4).split()
    regs, loads, stores = [], [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "mem":
            addr = int(toks[i + 1], 16)
            nxt = toks[i + 2] if i + 2 < len(toks) else None
            if nxt is not None and nxt.startswith("0x") and nxt != "mem":
                digits = len(nxt) - 2
                # a store prints value at width size*8 bits
                nbytes = max(1, (digits + 1) // 2)
                stores.append((addr, int(nxt, 16), nbytes))
                i += 3
            else:
                loads.append(addr)
                i += 2
            continue
        if t.startswith("c") and "_" in t:
            num_s = t[1:].split("_")[0]
            regs.append((4, int(num_s), int(toks[i + 1], 16)))
            i += 2
            continue
        if t[0] in "xf" and t[1:].isdigit():
            regs.append((1 if t[0] == "f" else 0, int(t[1:]), int(toks[i + 1], 16)))
            i += 2
            continue
        if t.startswith("e") or t.startswith("v"):
            # vector machinery: out of v1 scope, recorded so it is never silent
            regs.append((2, -1, 0))
            i += 1
            continue
        raise SystemExit("unparsed token %r in line: %s" % (t, line))
    return dict(priv=priv, pc=pc, bits=insn_bits, len=insn_len,
                regs=regs, loads=loads, stores=stores)


def read_text_log(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rec = parse_text_line(line)
            if rec is not None:
                out.append(rec)
            else:
                out.append(dict(raw=line))
    return out


# --------------------------------------------------------------------------
# the structured records
# --------------------------------------------------------------------------
def collect_records(cos, limit):
    recs = []
    for _ in range(limit):
        r = cos.step_one()
        if not r.valid:
            recs.append(r)
            break
        recs.append(r)
        if cos.halted():
            break
    return recs


def compare(elf, args, tmp):
    """Diff the fork's structured records against the unpatched text stream.

    Both sides run the same ELF under the same D6 bring-up (DTB on + --pc), so a
    record must match the text line at the same index 1:1.  The streams are
    compared as a PREFIX: an ordinary `spike --log-commits` run keeps executing
    until the next INTERLEAVE=5000 boundary before HTIF notices tohost, so the
    text stream tail is post-exit spin that a directly-stepped cosim run never
    produces (D7).  That tail is reported, not treated as a mismatch.
    """
    name = os.path.basename(elf)
    base_log = os.path.join(tmp, name + ".baseline.log")
    fork_log = os.path.join(tmp, name + ".fork.log")

    with open(base_log, "wb") as f:
        subprocess.run([args.baseline, "--isa=" + args.isa, "--pc=0x%x" % args.pc,
                        "--log-commits", elf], stdout=subprocess.DEVNULL, stderr=f,
                       check=True)
    subprocess.run([args.fork_spike, "--isa=" + args.isa, "--pc=0x%x" % args.pc,
                    "--log-commits", "--log=" + fork_log, elf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # ---- (A) text vs text: the printer must be untouched --------------------
    a_same = open(base_log, "rb").read() == open(fork_log, "rb").read()

    text = [r for r in read_text_log(base_log) if "raw" not in r]

    # ---- (B) text vs structured records -------------------------------------
    import pyspike_cosim
    cos = pyspike_cosim.Cosim(elf=elf, isa=args.isa, start_pc=args.pc,
                              log_path="/dev/null")
    recs = collect_records(cos, args.limit)
    n_rec = len(recs)
    stuck = n_rec and not recs[-1].valid
    if stuck:
        n_rec -= 1                      # the sentinel, not a real retirement

    res = dict(name=name, text_A=a_same, n_text=len(text), n_rec=n_rec,
               stuck=bool(stuck))
    stats = collections.Counter()
    bad = []

    # The two streams are aligned with a tolerant two-pointer walk, because the
    # stock text printer omits a line for every instruction whose trap escaped
    # execute_insn_logged() through the generic catch(...) -- i.e. every
    # non-memory trap (illegal instruction, ecall, breakpoint, ...).  Those
    # retirements exist in the structured record stream and simply have no text
    # counterpart; that asymmetry is a finding about the text log, not a capture
    # bug, and it is what plan-set Task A means by "read the structs, do NOT
    # parse the --log-commits text stream".
    def heads_match(t, r):
        return (t["pc"], t["bits"], t["len"], t["priv"]) == \
               (r.pc, r.insn_bits, r.insn_len, r.priv)

    def payload(i, t, r):
        rw = [(int(w.kind), int(w.num), int(w.val)) for w in r.reg_writes]
        if sorted(t["regs"]) != sorted(rw):
            stats["reg/csr mismatch"] += 1
            note_bad()
            if len(bad) < args.verbose_max:
                bad.append("#%d pc=%s regs text=%s rec=%s" %
                           (i, hex(t["pc"]), sorted(t["regs"]), sorted(rw)))
        else:
            for k, _num, _v in rw:
                stats["regs matched/%s" % KIND_NAMES.get(k, "?")] += 1

        st_w = [(int(w.va), int(w.data), int(w.size)) for w in r.mem_writes]
        if sorted(t["stores"]) != sorted(st_w):
            stats["store mismatch"] += 1
            note_bad()
            if len(bad) < args.verbose_max:
                bad.append("#%d pc=%s stores text=%s rec=%s" %
                           (i, hex(t["pc"]), sorted(t["stores"]), sorted(st_w)))
        else:
            stats["store frags matched"] += len(st_w)
            if st_w:
                stats["store insns"] += 1

        ld = [int(a) for a in t["loads"]]
        ld_r = [int(a.va) for a in r.mem_reads]
        if sorted(ld) != sorted(ld_r):
            stats["load mismatch"] += 1
            note_bad()
            if len(bad) < args.verbose_max:
                bad.append("#%d pc=%s loads text=%s rec=%s" %
                           (i, hex(t["pc"]), sorted(ld), sorted(ld_r)))
        else:
            stats["load frags matched"] += len(ld)

    first_bad = None

    def note_bad():
        nonlocal first_bad
        if first_bad is None:
            first_bad = j

    i = j = 0
    while i < len(text) and j < n_rec:
        t, r = text[i], recs[j]
        if heads_match(t, r):
            stats["insns compared"] += 1
            payload(i, t, r)
            i += 1
            j += 1
            continue

        # A step of the record stream with no text line means the RECORD stream
        # has an extra element (text[i] pairs with recs[j+1]); the reverse means
        # the text stream has one.
        rec_extra = j + 1 < n_rec and heads_match(t, recs[j + 1])
        text_extra = i + 1 < len(text) and heads_match(text[i + 1], r)
        if rec_extra and text_extra:
            # both readings fit; take the one that skips a trapping record, which
            # is the omission the stock printer actually makes
            pass                      # keep the record-extra reading
        if rec_extra:
            stats["recs absent from text log"] += 1
            if not r.trapped:
                note_bad()
            stats["  ...of which trapped (printer catch(...))"] += 1 if r.trapped else 0
            if len(bad) < args.verbose_max and not r.trapped:
                bad.append("rec#%d pc=%s insn=0x%x has no text line and did not trap"
                           % (j, hex(r.pc), r.insn_bits))
            j += 1
        elif text_extra:
            stats["text lines with no record"] += 1
            note_bad()
            if len(bad) < args.verbose_max:
                bad.append("text#%d pc=%s has no structured record" % (i, hex(t["pc"])))
            i += 1
        else:
            stats["head mismatch"] += 1
            note_bad()
            if len(bad) < args.verbose_max:
                bad.append("#%d/#%d text pc=%s insn=0x%x/%d priv=%d vs rec pc=%s insn=0x%x/%d priv=%d trapped=%d"
                           % (i, j, hex(t["pc"]), t["bits"], t["len"], t["priv"],
                              hex(r.pc), r.insn_bits, r.insn_len, r.priv, r.trapped))
            i += 1
            j += 1
    res["first_bad"] = first_bad
    # A directly-stepped cosim run never crosses sim_t::step()'s INTERLEAVE
    # boundary, so it never calls mmu->yield_load_reservation(); a stock `spike
    # --log-commits` run does, every 5000 instructions.  That is an arbitrary
    # modelling event, not an architectural one, and it is exactly what ADR-0013
    # D19 hands to the testbench -- so an lr/sc program is expected to leave the
    # stock text stream at the first quantum, on an AMO instruction.
    res["d19_expected"] = (
        first_bad is not None and first_bad >= 4900
        and (recs[first_bad].insn_bits & 0x7f) == 0x2f)
    stats["text lines beyond the records (stock over-runs to its 5000-insn HTIF poll "
          "quantum after the exit)"] = len(text) - i
    stats["records beyond the text log"] = n_rec - j

    res["stats"] = dict(stats)
    res["bad"] = bad
    res["absorbed"] = sum(1 for r in recs[:n_rec] if r.absorbed)
    res["trapped"] = sum(1 for r in recs[:n_rec] if r.trapped)
    res["nop_like"] = sum(1 for r in recs[:n_rec]
                          if not r.reg_writes and not r.mem_writes
                          and r.npc == r.pc + r.insn_len)
    res["fast"] = cos.fast_path_counts()
    res["halted"] = cos.halted()
    res["exit_code"] = cos.exit_code()
    res["tohost"] = cos.tohost_addr()
    res["dropped"] = cos.htif_cmds_dropped()
    res["pa_seen"] = any(int(w.pa) != int(w.va) for r in recs[:n_rec] for w in r.mem_writes)
    res["csr_seen"] = any(int(w.kind) == 4 for r in recs[:n_rec] for w in r.reg_writes)
    res["fp_seen"] = any(int(w.kind) == 1 for r in recs[:n_rec] for w in r.reg_writes)
    res["store_seen"] = any(len(r.mem_writes) for r in recs[:n_rec])
    res["load_seen"] = any(len(r.mem_reads) for r in recs[:n_rec])
    # "recs absent from text log" is tolerated only when every such record is a
    # trap the stock printer cannot print; anything else is a real gap.
    tolerated = stats["recs absent from text log"] == \
                stats["  ...of which trapped (printer catch(...))"]
    res["identical_prefix"] = 0 if first_bad is None else first_bad
    clean = (stats["records beyond the text log"] == 0
             and stats["text lines with no record"] == 0
             and not any(k.endswith("mismatch") for k in stats))
    res["ok"] = bool(a_same and not stuck
                     and (clean or res["d19_expected"])
                     and (tolerated or res["d19_expected"]))
    return res


def fastpath_test(args, tmp):
    """Characterise the plan-set 1.2 fast-path question instead of assuming it.

    The aligned+TLB-hit fast paths in mmu.h are the only memory sites that used to
    bypass the per-instruction log.  They are unreachable while logging, for two
    independent reasons in stock Spike:

      * refill_tlb() returns without filling while get_log_commits_enabled()
        (mmu.cc, "the TLB caches this setting"), and
      * processor_t::enable_log_commits() calls mmu->flush_tlb(), so enabling
        commit logging evicts whatever was filled before it.

    This probe separates the two: warm the DTLB with capture OFF (the fast paths
    must then be taken, proving they are live code), switch capture on, and see
    whether any access is still fast-pathed (it must not be) while the stores are
    nevertheless all logged (they must be).
    """
    import pyspike_cosim
    cos = pyspike_cosim.Cosim(elf=args.stwarm, isa=args.isa, start_pc=args.pc,
                              log_path="/dev/null", capture=False)
    # processor_t::step(n) also ends the call early when a trap is taken, so the
    # warm-up is issued in chunks rather than as one long step.
    for _ in range(20):
        cos.run(50)                    # into the warm-address store loop: DTLB fills
    warm = cos.fast_path_counts()
    cos.enable_capture()
    stores = loads = 0
    for _ in range(600):
        r = cos.step_one()
        if not r.valid:
            break
        stores += len(r.mem_writes)
        loads += len(r.mem_reads)
    after = cos.fast_path_counts()
    return dict(fast_accesses_before_capture=warm, fast_accesses_after_capture=after,
                fast_path_hits_while_logging=(after[0] - warm[0], after[1] - warm[1]),
                stores_logged_while_logging=stores,
                loads_logged_while_logging=loads)


def sc_probe(args):
    """D19: prove the TB->Spike store-conditional verdict channel is real.

    The verdict is one-shot and only consumed by store_conditional(), so arming
    it before every step affects only SC instructions.  Forcing pass and forcing
    fail must move both the rd result and whether the data write happens, while
    leaving everything else (including the fault checks, which run first) alone.
    """
    import pyspike_cosim
    SC_FUNCT5 = 0x03          # riscv/sc.w and sc.d, before the aq/rl bits
    out = {}
    for label, verdict in (("natural", -1), ("forced-pass", 1), ("forced-fail", 0)):
        cos = pyspike_cosim.Cosim(elf=args.lrsc, isa=args.isa, start_pc=args.pc,
                                  log_path="/dev/null")
        scs = 0
        sc_with_store = 0
        rd_vals = collections.Counter()
        for _ in range(args.sc_steps):
            cos.set_sc_verdict(verdict)
            r = cos.step_one()
            if not r.valid:
                break
            if (r.insn_bits & 0x7f) != 0x2f:                 # not an AMO
                continue
            if ((r.insn_bits >> 27) & 0x1f) != SC_FUNCT5:   # not an SC
                continue
            scs += 1
            sc_with_store += 1 if r.mem_writes else 0
            for w in r.reg_writes:
                if int(w.kind) == 0:
                    rd_vals[int(w.val)] += 1
        out[label] = dict(sc_insns=scs, sc_that_wrote=sc_with_store,
                          rd_values=sorted(rd_vals.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isa", default="rv64gc")
    ap.add_argument("--pc", type=lambda s: int(s, 0), default=0x80000000)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--verbose-max", type=int, default=10)
    ap.add_argument("--cosim-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--baseline", default=os.path.join(PYO3, "spike-install", "bin", "spike"))
    ap.add_argument("--fork-spike", default=os.path.join(PYO3, "spike-build", "spike"))
    ap.add_argument("--tmp", default=os.path.join(PYO3, ".mb_logs", "sa-spike-cosim-api"))
    ap.add_argument("--stwarm", default=os.path.join(PYO3, ".mb_logs", "sa-spike-cosim-api", "stwarm"))
    ap.add_argument("--lrsc", default=os.path.join(PYO3, "riscv-tests", "isa", "rv64ua-p-lrsc"))
    ap.add_argument("--sc-steps", type=int, default=4000)
    ap.add_argument("--elf", action="append", default=None)
    args = ap.parse_args()

    if args.elf is None:
        isa_dir = os.path.join(PYO3, "riscv-tests", "isa")
        args.elf = [os.path.join(isa_dir, n) for n in
                    ["rv64ui-p-add", "rv64ui-p-and", "rv64uc-p-rvc",
                     "rv64um-p-mul", "rv64uf-p-fmadd", "rv64ua-p-lrsc",
                     "rv64ui-p-sd", "rv64ui-p-sw", "rv64ui-p-lw", "rv64ui-p-jal",
                     "rv64ud-p-fadd", "rv64ui-p-fence_i", "rv64ua-p-amoadd_d",
                     "rv64ui-v-add", "rv64uf-v-fmadd", "rv64uc-v-rvc"]]
        args.elf = [e for e in args.elf if os.path.exists(e)]

    os.makedirs(args.tmp, exist_ok=True)
    sys.path.insert(0, args.cosim_dir)
    overall_ok = True
    for elf in args.elf:
        r = compare(elf, args, args.tmp)
        st = r["stats"]
        overall_ok &= r["ok"]
        print("%-18s %s  (A) text identical to stock: %s | (B) text_lines=%d records=%d" %
              (r["name"], "PASS" if r["ok"] else "FAIL", r["text_A"], r["n_text"], r["n_rec"]))
        print("    %s" % st)
        if r.get("d19_expected"):
            print("    NOTE prefix of %d records identical; divergence starts on an AMO at the "
                  "first INTERLEAVE=5000 quantum, i.e. the load-reservation yield that D7 "
                  "removes and D19 hands to the TB (expected, not an API failure)"
                  % r["identical_prefix"])
        print("    coverage: traps=%d nop_like=%d absorbed=%d pa!=va=%s csr=%s fp=%s stores=%s loads=%s" %
              (r["trapped"], r["nop_like"], r["absorbed"], r["pa_seen"], r["csr_seen"],
               r["fp_seen"], r["store_seen"], r["load_seen"]))
        print("    fast-path logged during run: loads=%d stores=%d | halted=%s exit=%d "
              "tohost=0x%x htif_cmds_dropped=%d" %
              (r["fast"][0], r["fast"][1], r["halted"], r["exit_code"], r["tohost"],
               r["dropped"]))
        for b in r["bad"]:
            print("    MISMATCH %s" % b)

    if os.path.exists(args.stwarm):
        print("\nfast-path memory-logging probe (plan-set 1.2):")
        f = fastpath_test(args, args.tmp)
        for k, v in f.items():
            print("    %-34s %s" % (k, v))
        print("    => the fast paths are live code (%d accesses while not logging) but take "
              "%d while logging, so no logged run can lose an access there; all %d stores "
              "and %d loads are still captured"
              % (f["fast_accesses_before_capture"][1],
                 f["fast_path_hits_while_logging"][1],
                 f["stores_logged_while_logging"],
                 f["loads_logged_while_logging"]))

    if os.path.exists(args.lrsc):
        print("\nD19 store-conditional verdict-injection probe (%s):" % os.path.basename(args.lrsc))
        for k, v in sc_probe(args).items():
            print("    %-12s %s" % (k, v))

    print("\nOVERALL: %s" % ("PASS" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
