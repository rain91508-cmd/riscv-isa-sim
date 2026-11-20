// See LICENSE for license details.

// Structured single-retirement step API for lockstep co-simulation with the
// gem5 PyO3 CL model.  See ADR-0013 and docs/superpowers/plans/
// 2026-09-11-plan-set-spike-cosim.md (D12/D17/D18/D19).
//
// This file only *reads out* state Spike already maintains.  It adds no ISA
// behaviour and changes no logging semantics: the write sets come from the same
// state_t::log_* buffers that --log-commits prints, sampled at the point where
// they are complete for one instruction.

#include "config.h"
#include "cosim.h"
#include "processor.h"
#include "mmu.h"

#include <cassert>
#include <utility>

// cosim_step_one() absorbs steps that retire nothing.  processor_t::step(1)
// already loops internally past PC_SERIALIZE_BEFORE no-ops (its inner
// `while (instret < n)` only ends on a retirement), so this guard exists purely
// for the paths that return without retiring -- WFI re-entry, a debug-mode
// break, single-step.  Hitting the guard is reported to the caller as
// valid == false rather than looping forever.
static const size_t COSIM_ABSORB_GUARD = 1000;

void processor_t::cosim_enable_capture()
{
  // log_commits_enabled() is the gate on every capture site: the per-inscription
  // buffer reset in execute_insn_logged(), csr_t::log_special_write(), and the
  // mmu log pushes.  Turning it on is what turns cosim capture on.
  enable_log_commits();
}

void processor_t::cosim_set_sc_verdict(int verdict)
{
  mmu->cosim_sc_verdict = verdict;
}

std::pair<uint64_t, uint64_t> processor_t::cosim_fast_path_counts() const
{
  return {mmu->cosim_fast_loads, mmu->cosim_fast_stores};
}

StepRecord processor_t::cosim_step_one()
{
  StepRecord rec;

  if (!get_log_commits_enabled())
    return rec;  // valid == false: capture was never enabled

  state_t* st = &state;
  const uint64_t retires_before = st->cosim.retire_count;

  for (size_t attempt = 0; attempt < COSIM_ABSORB_GUARD; ++attempt) {
    st->cosim.trap_valid = false;
    st->cosim.attempt_stashed = false;
    // P7/M1 (ADR-0013 D12 record contract): clear the write logs at the
    // start of EVERY step attempt, so a record can never carry another
    // step's writes.  This matters for FETCH-SIDE traps: mmu->load_insn()
    // throws before execute_insn_logged() gets to re-reset, so without this
    // clear the trap record would pick up the previous instruction's stale
    // log buffers.  Execute-side traps are unaffected: their own attempt
    // reaches execute_insn_logged(), which re-resets and then keeps the
    // attempt's legitimate partial writes + take_trap()'s CSR writes.
    st->log_reg_write.clear();
    st->log_mem_read.clear();
    st->log_mem_write.clear();
    step(1);
    if (st->cosim.retire_count != retires_before || st->cosim.trap_valid) {
      rec.absorbed = (uint32_t)attempt;
      break;
    }
    if (attempt + 1 == COSIM_ABSORB_GUARD)
      return rec;  // valid == false: the hart is not retiring (WFI / debug)
  }

  rec.valid      = true;
  rec.retired    = (st->cosim.retire_count != retires_before);
  rec.trapped    = st->cosim.trap_valid;
  rec.pc         = st->cosim.insn_pc;
  rec.insn_bits  = st->cosim.insn_bits;
  rec.insn_len   = st->cosim.insn_len;
  rec.npc        = st->pc;
  rec.priv       = (uint32_t)st->last_inst_priv;
  rec.xlen       = (uint32_t)st->last_inst_xlen;
  rec.flen       = (uint32_t)st->last_inst_flen;
  rec.prv_changed = st->prv_changed;
  rec.retire_seq = st->cosim.retire_count;
  rec.commit_no  = ++st->cosim.records;

  if (rec.trapped && !st->cosim.attempt_stashed) {
    // P7/M1 fetch-side trap: the trap fired in mmu->load_insn() BEFORE
    // execute_insn_logged() could stash this step's identity, so the fields
    // above still hold the PREVIOUS instruction's identity (the bug this
    // replaces: a fetch trap reported as `pc=0x80000150 SRET`).  The record
    // is rebuilt with the trap's own facts: pc = trap_epc, the explicit
    // NO-ENCODING marker (insn_bits = 0, insn_len = 0 -- a fetch-side trap
    // composite has no encoding), and priv = privilege at the fault (not the
    // stale last_inst_priv).  Capture-only; no ISA semantics change.
    rec.pc        = st->cosim.trap_epc;
    rec.insn_bits = 0;
    rec.insn_len  = 0;
    rec.priv      = st->cosim.trap_priv;
  }

  if (rec.trapped) {
    rec.cause = st->cosim.trap_cause;
    rec.tval  = st->cosim.trap_tval;
    rec.tval2 = st->cosim.trap_tval2;
    rec.tinst = st->cosim.trap_tinst;
    rec.epc   = st->cosim.trap_epc;
  }

  // Register / CSR writes.  Key encoding is (num << 4) | kind, kind 0=int,
  // 1=fp, 2=vec, 3=vec element, 4=CSR (see commit_log_print_insn).  The value
  // is stored unconverted, which is what D10 asks for: raw 64-bit FP, no
  // NaN-box canonicalisation, fflags as its own CSR write.
  for (const auto& item : st->log_reg_write) {
    const reg_t key = item.first;
    // x0 is architecturally inert; the text printer skips it too, so skipping
    // it here keeps the structured stream and the text stream interchangeable.
    if (key == 0)
      continue;
    CosimRegWrite w;
    w.kind    = (uint8_t)(key & 0xf);
    w.num     = (uint32_t)(key >> 4);
    w.val     = item.second.v[0];
    w.val_hi  = item.second.v[1];
    rec.reg_writes.push_back(w);
  }

  for (const auto& item : st->log_mem_write)
    rec.mem_writes.push_back(CosimMemWrite{(uint64_t)std::get<0>(item),
                                           (uint64_t)std::get<1>(item),
                                           (uint32_t)std::get<3>(item),
                                           std::get<2>(item)});

  // Loads carry no value: Spike's log only records that an address was read,
  // so a bad load is proven through its destination-register write above.
  for (const auto& item : st->log_mem_read)
    rec.mem_reads.push_back(CosimMemRead{(uint64_t)std::get<0>(item),
                                         (uint64_t)std::get<1>(item),
                                         (uint32_t)std::get<3>(item)});

  return rec;
}
