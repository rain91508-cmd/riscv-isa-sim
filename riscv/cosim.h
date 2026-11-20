// See LICENSE for license details.
#ifndef _RISCV_COSIM_H
#define _RISCV_COSIM_H

// ---------------------------------------------------------------------------
// Structured per-retired-instruction step record for lockstep co-simulation
// (ADR-0013, plan-set 2026-09-11 D12/D17/D18/D19).
//
// Everything here is *capture only*: the write sets are read out of Spike's
// existing per-instruction log buffers (state_t::log_reg_write /
// log_mem_read / log_mem_write) at the same point the --log-commits printer
// consumes them.  No ISA semantics, no logging semantics are changed.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

// Kind nibble of a log_reg_write key: key == (num << 4) | kind.
// Mirrors the switch in commit_log_print_insn(); kept as plain integers so the
// binding does not have to agree with Spike on an enum type.
enum cosim_reg_kind {
  COSIM_REG_INT      = 0,
  COSIM_REG_FP       = 1,
  COSIM_REG_VEC      = 2,
  COSIM_REG_VECELEM  = 3,
  COSIM_REG_CSR      = 4,
};

// One architectural register / CSR write made by the retiring instruction.
struct CosimRegWrite {
  uint8_t  kind = 0;    // cosim_reg_kind
  uint32_t num  = 0;    // register number, or CSR address when kind == CSR
  uint64_t val  = 0;    // raw low 64 bits (int/CSR: xlen-wide; FP: raw, unboxed)
  uint64_t val_hi = 0;  // high 64 bits, non-zero only for Q/V writes
};

// One memory store fragment (Spike fragments stores to <= 8 bytes).
struct CosimMemWrite {
  uint64_t va   = 0;    // virtual address as the instruction saw it
  uint64_t pa   = 0;    // physical address after translation (D17)
  uint32_t size = 0;    // bytes in this fragment
  uint64_t data = 0;    // little-endian value written
};

// One memory load access.  Spike never records the loaded *value*, so load
// correctness is proven through the destination-register write (D10/plan 1.1).
struct CosimMemRead {
  uint64_t va   = 0;
  uint64_t pa   = 0;
  uint32_t size = 0;
};

// The complete architectural delta of ONE genuinely-retired instruction.
struct StepRecord {
  bool     valid   = false;  // a record was produced (false => stuck/no retire)
  bool     retired = false;  // an architectural instruction retired
  bool     trapped = false;  // take_trap() ran during this step (D8)

  uint64_t pc        = 0;    // pc of the retiring/faulting instruction
  uint64_t npc       = 0;    // state.pc after the step (fall-through or tvec)
  uint64_t insn_bits = 0;    // raw instruction encoding
  uint32_t insn_len  = 0;    // instruction length in bytes (2 or 4)

  uint32_t priv = 0;         // privilege of the retiring instruction
  uint32_t xlen = 64;
  uint32_t flen = 0;
  bool     prv_changed = false;

  std::vector<CosimRegWrite> reg_writes;
  std::vector<CosimMemWrite> mem_writes;
  std::vector<CosimMemRead>  mem_reads;

  // Valid when trapped.  Per D8 the faulting instruction must show NO dest or
  // memory writes; the trap-CSR writes appear in reg_writes of this record.
  uint64_t cause = 0;
  uint64_t tval  = 0;
  uint64_t tval2 = 0;
  uint64_t tinst = 0;
  uint64_t epc   = 0;

  // Diagnostics for D11/D12: how many extra processor_t::step(1) calls this
  // record needed beyond the first (serialize/WFI absorption), the value of
  // Spike's retirement counter after this step, and a 1-based record number.
  uint32_t absorbed   = 0;
  uint64_t retire_seq = 0;
  uint64_t commit_no  = 0;
};

// Per-processor capture scratch, living in state_t so that the capture points
// inside execute.cc / processor.cc can reach it without a global.
struct cosim_capture_t {
  // Stashed at the top of every logged instruction attempt.
  uint64_t insn_pc   = 0;
  uint64_t insn_bits = 0;
  uint32_t insn_len  = 0;
  // Bumped wherever processor_t::step()'s local `instret` is bumped, i.e. on
  // every genuine retirement.  Lets cosim_step_one() tell a retirement from a
  // serialized/WFI no-op step.
  uint64_t retire_count = 0;
  // Written by take_trap(); cleared by cosim_step_one() before each step.
  bool     trap_valid = false;
  uint64_t trap_cause = 0, trap_tval = 0, trap_tval2 = 0, trap_tinst = 0;
  uint64_t trap_epc   = 0;
  // Privilege AT the fault (state.prv before take_trap() delivers the trap),
  // captured with the trap fields.  A fetch-side trap record has no
  // last_inst_priv of its own (the attempt never reached
  // execute_insn_logged()), so the stale one must not be reported.
  uint32_t trap_priv  = 0;
  // Set by commit_log_stash_insn() when one logged instruction attempt has
  // stashed its identity this step; cleared by cosim_step_one() at the start
  // of EVERY step attempt.  A trap with no stashed attempt is a FETCH-SIDE
  // trap (thrown by mmu->load_insn() before execute_insn_logged()) and gets
  // the explicit no-encoding record (pc=trap_epc, insn=0, len=0).
  bool     attempt_stashed = false;
  // Incremented by cosim_step_one() for each record handed to the caller.
  uint64_t records = 0;
};

#endif // _RISCV_COSIM_H
