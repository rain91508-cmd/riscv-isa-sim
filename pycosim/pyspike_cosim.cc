// See LICENSE for license details.
//
// pyspike_cosim -- pybind11 extension exposing Spike's structured single-step
// co-simulation API to CPython (ADR-0013, plan-set 2026-09-11 Task A).
//
// Bring-up follows D6: a real sim_t with the DTB ENABLED (so the DTS parse
// runs and set_max_vaddr_bits(39) happens, which every -v- Sv39 test needs)
// plus cfg.start_pc set to the ELF entry, so no boot-ROM instruction is
// executed.  Stepping follows D7: processor_t::step(1) is driven directly
// through cosim_step_one(); sim_t::step()/run() are never called, so the
// yield_load_reservation() and devices->tick() side effects that live at the
// INTERLEAVE=5000 boundary cannot fire mid-compare.  HTIF's tohost is polled
// explicitly instead (see halted()).
//
// The module is imported only when co-simulation is enabled (D21); nothing in
// the CL regression path touches it.

// <sys/syscall.h> first: libstdc++'s <atomic_wait.h>, pulled in transitively by
// pybind11 through <thread>, names SYS_futex and is only satisfied once
// <sys/syscall.h> has been seen.  Spike's own TU graph happens to pull it in
// before <thread>; this binding does not, so include it up front.
#include <sys/syscall.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "cosim.h"
#include "processor.h"
#include "mmu.h"
#include "sim.h"
#include "devices.h"
#include "platform.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class Cosim {
public:
  // Declaration order is construction order and the reverse is destruction
  // order: the strings and cfg_t must outlive sim_t (which keeps a const
  // cfg_t*), and sim_t must be destroyed before we free the memories it wired
  // into its bus (sim_t does not own them -- see spike.cc).
  std::string isa_s;
  std::string priv_s;
  std::string log_path_s;
  cfg_t       cfg;
  std::vector<std::pair<reg_t, abstract_mem_t*>> mems;
  std::unique_ptr<sim_t> sim;
  processor_t* proc = nullptr;
  bool tohost_seen = false;
  uint64_t tohost_exit_code = 0;
  uint64_t htif_cmds_dropped = 0;
  uint64_t steps = 0;

  Cosim(const std::string& elf,
        const std::string& isa      = "rv64gc",
        uint64_t start_pc           = DRAM_BASE,
        const std::string& priv     = "",
        bool dtb                    = true,
        const std::string& log_path = "/dev/null",
        uint64_t mem_base           = 0,
        uint64_t mem_bytes          = 0,
        bool     capture            = true)
    : isa_s(isa), priv_s(priv), log_path_s(log_path)
  {
    cfg.isa  = isa_s.c_str();
    // empty priv keeps cfg_t's own default (DEFAULT_PRIV, "MSU"); spike
    // validates this string, so do not invent a value for it.
    if (!priv_s.empty())
      cfg.priv = priv_s.c_str();
    if (mem_bytes)
      cfg.mem_layout = std::vector<mem_cfg_t>{mem_cfg_t(reg_t(mem_base), (size_t)mem_bytes)};
    cfg.start_pc.set_global(reg_t(start_pc));

    mems.reserve(cfg.mem_layout.size());
    for (auto& m : cfg.mem_layout)
      mems.push_back(std::make_pair(m.get_base(), new mem_t(m.get_size())));

    const std::vector<device_factory_sargs_t> no_devices;
    const std::vector<std::string> htif_args{elf};
    debug_module_config_t dm_config;

    try {
      sim = std::make_unique<sim_t>(&cfg, /*halted*/false, mems, no_devices,
                                    /*dtb_discovery*/false, htif_args, dm_config,
                                    log_path_s.c_str(), /*dtb_enabled*/dtb,
                                    /*dtb_file*/nullptr, /*socket_enabled*/false,
                                    /*cmd_file*/nullptr, /*instruction_limit*/std::nullopt);
    } catch (const std::exception& e) {
      for (auto& m : mems) delete m.second;
      mems.clear();
      throw std::runtime_error(std::string("spike sim_t bring-up failed: ") + e.what());
    }

    if (cfg.nprocs() < 1) {
      for (auto& m : mems) delete m.second;
      mems.clear();
      throw std::runtime_error("spike config has no harts");
    }
    proc = sim->get_core(0);
    // htif_t needs the target xlen to interpret tohost words.
    sim->set_expected_xlen(proc->get_isa().get_max_xlen());
    // htif_t::start() is what loads the payload ELF into target memory and then
    // calls sim_t::reset(), which re-applies cfg->start_pc.  Normally it runs
    // from inside htif_t::run(), which D7 forbids -- so the bring-up step has to
    // be invoked on its own here, otherwise the machine boots over empty RAM.
    sim->start();
    if (capture)
      proc->cosim_enable_capture();
  }

  // Start capturing (or re-start after a warm-up).  Passing capture=false to
  // the constructor and then running step_one-free warm-up steps leaves DTLB
  // entries filled -- which is the only way to reach the aligned+TLB-hit MMU
  // fast paths at all while logging, so this is also the hook the fast-path
  // store-coverage test uses.
  void enable_capture() { proc->cosim_enable_capture(); }

  // Free-run the hart n instructions with no record produced.  Uses
  // processor_t::step() directly, never sim_t::step() (D7).
  void run(uint64_t n) { proc->step((size_t)n); }

  py::tuple fast_path_counts()
  {
    auto c = proc->cosim_fast_path_counts();
    return py::make_tuple(c.first, c.second);
  }

  ~Cosim()
  {
    sim.reset();               // stops the bus referencing the memories
    for (auto& m : mems) delete m.second;
    mems.clear();
  }

  Cosim(const Cosim&) = delete;
  Cosim& operator=(const Cosim&) = delete;

  StepRecord step_one()
  {
    ++steps;
    return proc->cosim_step_one();
  }

  void set_sc_verdict(int verdict)
  {
    if (verdict != -1 && verdict != 0 && verdict != 1)
      throw std::invalid_argument("sc verdict must be -1 (no override), 0 (fail) or 1 (pass)");
    proc->cosim_set_sc_verdict(verdict);
  }

  // HTIF polls tohost only inside htif_t::run(), which D7 forbids, so the
  // verdict has to be pulled here.  Same protocol, same decode (>>1).
  void poll_tohost()
  {
    if (tohost_seen)
      return;
    const addr_t th = sim->get_tohost_addr();
    if (th == 0)
      return;                    // ELF has no tohost symbol (e.g. -v- only)
    uint64_t v = 0;
    try {
      sim->memif().read(th, sizeof(v), &v);
    } catch (...) {
      return;
    }
    if (v == 0)
      return;
    const uint64_t raw = v;      // little-endian target
    sim->memif().write(th, sizeof(v), &zero64);
    // command_t decodes device = tohost >> 56 and cmd = tohost >> 48, so a word
    // with those bytes clear is the classic terminate form (code << 1) | 1.
    if ((raw >> 48) == 0 && (raw & 1)) {
      tohost_seen = true;
      tohost_exit_code = raw >> 1;
    } else {
      // A device command (console, syscall, ...).  Servicing those needs
      // htif_t::run()'s device list, which D7 keeps out of cosim; count them so
      // a driver can tell a dropped command from a real end of run.
      ++htif_cmds_dropped;
    }
  }
  static constexpr uint64_t zero64 = 0;

  bool halted()
  {
    if (sim->done())
      return true;
    poll_tohost();
    return tohost_seen || proc->get_state()->critical_error;
  }

  int exit_code()
  {
    if (sim->done())
      return sim->exit_code();
    poll_tohost();
    return tohost_seen ? (int)tohost_exit_code : (proc->get_state()->critical_error ? -1 : 0);
  }

  py::bytes read_mem(uint64_t paddr, uint64_t len)
  {
    if (len == 0 || len > (1u << 24))
      throw std::invalid_argument("read_mem: bad length");
    std::string buf(len, '\0');
    sim->memif().read(addr_t(paddr), (size_t)len, &buf[0]);
    return py::bytes(buf);
  }

  uint64_t mem_digest(uint64_t paddr, uint64_t len, uint64_t chunk)
  {
    if (!chunk || chunk > (1u << 20))
      chunk = 1u << 16;
    uint64_t h = 1469598103934665603ull;          // FNV-1a 64
    std::string buf;
    for (uint64_t off = 0; off < len; off += chunk) {
      uint64_t n = std::min<uint64_t>(chunk, len - off);
      buf.assign((size_t)n, '\0');
      sim->memif().read(addr_t(paddr + off), (size_t)n, &buf[0]);
      for (unsigned char c : buf) { h ^= c; h *= 1099511628211ull; }
    }
    return h;
  }

  uint64_t pc()    const { return (uint64_t)proc->get_state()->pc; }
  uint64_t prv()   const { return (uint64_t)proc->get_state()->prv; }
  bool     virt()  const { return proc->get_state()->v; }
  uint64_t get_xpr(uint64_t i) const { return (uint64_t)proc->get_state()->XPR[(size_t)i]; }
  uint64_t get_fpr(uint64_t i) const { return proc->get_state()->FPR[(size_t)i].v[0]; }
  uint64_t get_csr(uint64_t num)     { return (uint64_t)proc->get_csr((int)num); }
  uint64_t tohost_addr() const       { return (uint64_t)sim->get_tohost_addr(); }
  uint64_t dropped_cmds() const      { return htif_cmds_dropped; }
  uint64_t step_count() const        { return steps; }
  uint64_t entry_point() const       { return (uint64_t)cfg.start_pc.get(0).value_or(0); }
  uint64_t nprocs() const            { return (uint64_t)cfg.nprocs(); }
};

PYBIND11_MODULE(pyspike_cosim, m)
{
  m.doc() = "Spike structured single-retirement step API for O3 CL co-simulation (ADR-0013)";

  py::class_<CosimRegWrite>(m, "RegWrite")
    .def_readonly("kind",    &CosimRegWrite::kind)
    .def_readonly("num",     &CosimRegWrite::num)
    .def_readonly("val",     &CosimRegWrite::val)
    .def_readonly("val_hi",  &CosimRegWrite::val_hi)
    .def("__repr__", [](const CosimRegWrite& w) {
      return "<RegWrite kind=" + std::to_string((int)w.kind) + " num=" + std::to_string(w.num) +
             " val=0x" + [&]{ char b[32]; snprintf(b, sizeof b, "%016llx", (unsigned long long)w.val); return std::string(b); }() + ">";
    });

  py::class_<CosimMemWrite>(m, "MemWrite")
    .def_readonly("va",   &CosimMemWrite::va)
    .def_readonly("pa",   &CosimMemWrite::pa)
    .def_readonly("size", &CosimMemWrite::size)
    .def_readonly("data", &CosimMemWrite::data)
    .def("__repr__", [](const CosimMemWrite& w) {
      char b[96];
      snprintf(b, sizeof b, "<MemWrite va=0x%llx pa=0x%llx size=%u data=0x%llx>",
               (unsigned long long)w.va, (unsigned long long)w.pa, w.size, (unsigned long long)w.data);
      return std::string(b);
    });

  py::class_<CosimMemRead>(m, "MemRead")
    .def_readonly("va",   &CosimMemRead::va)
    .def_readonly("pa",   &CosimMemRead::pa)
    .def_readonly("size", &CosimMemRead::size)
    .def("__repr__", [](const CosimMemRead& r) {
      char b[64];
      snprintf(b, sizeof b, "<MemRead va=0x%llx pa=0x%llx size=%u>",
               (unsigned long long)r.va, (unsigned long long)r.pa, r.size);
      return std::string(b);
    });

  py::class_<StepRecord>(m, "StepRecord")
    .def_readonly("valid",       &StepRecord::valid)
    .def_readonly("retired",     &StepRecord::retired)
    .def_readonly("trapped",     &StepRecord::trapped)
    .def_readonly("pc",          &StepRecord::pc)
    .def_readonly("npc",         &StepRecord::npc)
    .def_readonly("insn_bits",   &StepRecord::insn_bits)
    .def_readonly("insn_len",    &StepRecord::insn_len)
    .def_readonly("priv",        &StepRecord::priv)
    .def_readonly("xlen",        &StepRecord::xlen)
    .def_readonly("flen",        &StepRecord::flen)
    .def_readonly("prv_changed", &StepRecord::prv_changed)
    .def_readonly("reg_writes",  &StepRecord::reg_writes)
    .def_readonly("mem_writes",  &StepRecord::mem_writes)
    .def_readonly("mem_reads",   &StepRecord::mem_reads)
    .def_readonly("cause",       &StepRecord::cause)
    .def_readonly("tval",        &StepRecord::tval)
    .def_readonly("tval2",       &StepRecord::tval2)
    .def_readonly("tinst",       &StepRecord::tinst)
    .def_readonly("epc",         &StepRecord::epc)
    .def_readonly("absorbed",    &StepRecord::absorbed)
    .def_readonly("retire_seq",  &StepRecord::retire_seq)
    .def_readonly("commit_no",   &StepRecord::commit_no)
    .def("__repr__", [](const StepRecord& r) {
      char b[160];
      snprintf(b, sizeof b, "<StepRecord %s pc=0x%llx insn=0x%llx len=%u priv=%u "
                            "regs=%zu stores=%zu loads=%zu%s%s absorbed=%u>",
               r.valid ? "valid" : "INVALID",
               (unsigned long long)r.pc, (unsigned long long)r.insn_bits, r.insn_len, r.priv,
               r.reg_writes.size(), r.mem_writes.size(), r.mem_reads.size(),
               r.retired ? " retired" : "", r.trapped ? " TRAPPED" : "", r.absorbed);
      return std::string(b);
    });

  py::class_<Cosim>(m, "Cosim")
    .def(py::init<const std::string&, const std::string&, uint64_t, const std::string&,
                  bool, const std::string&, uint64_t, uint64_t, bool>(),
         py::arg("elf"),
         py::arg("isa")       = "rv64gc",
         py::arg("start_pc")  = (uint64_t)DRAM_BASE,
         py::arg("priv")      = "",  // "" => spike's DEFAULT_PRIV
         py::arg("dtb")       = true,
         py::arg("log_path")  = "/dev/null",
         py::arg("mem_base")  = 0,
         py::arg("mem_bytes") = 0,
         py::arg("capture")   = true,
         "Bring up a spike machine for lockstep co-simulation (D6: DTB on + start_pc).")
    .def("step_one",           &Cosim::step_one,      "Step to the next genuinely retired instruction (D12).")
    .def("enable_capture",     &Cosim::enable_capture)
    .def("run",                &Cosim::run,           py::arg("n"))
    .def("fast_path_counts",   &Cosim::fast_path_counts,
         "(#loads, #stores) logged by the aligned+TLB-hit MMU fast paths")
    .def("set_sc_verdict",     &Cosim::set_sc_verdict, py::arg("verdict"),
         "Inject the TB's store-conditional verdict for the next SC (D19): 1 pass, 0 fail, -1 off.")
    .def("read_mem",           &Cosim::read_mem,      py::arg("paddr"), py::arg("len"))
    .def("mem_digest",         &Cosim::mem_digest,    py::arg("paddr"), py::arg("len"), py::arg("chunk") = 0)
    .def("halted",             &Cosim::halted)
    .def("exit_code",          &Cosim::exit_code)
    .def("poll_tohost",        &Cosim::poll_tohost)
    .def("htif_cmds_dropped",  &Cosim::dropped_cmds,
         "HTIF device commands seen on tohost but not serviced (D7 does not run htif_t::run)")
    .def("pc",                 &Cosim::pc)
    .def("prv",                &Cosim::prv)
    .def("virt",               &Cosim::virt)
    .def("get_xpr",            &Cosim::get_xpr,       py::arg("rs"))
    .def("get_fpr",            &Cosim::get_fpr,       py::arg("frs"))
    .def("get_csr",            &Cosim::get_csr,       py::arg("csr"))
    .def("tohost_addr",        &Cosim::tohost_addr)
    .def("step_count",         &Cosim::step_count)
    .def("entry_point",        &Cosim::entry_point)
    .def("nprocs",             &Cosim::nprocs);

  // Register-kind constants, matching Spike's log_reg_write key nibble.
  m.attr("REG_INT")     = (int)COSIM_REG_INT;
  m.attr("REG_FP")      = (int)COSIM_REG_FP;
  m.attr("REG_VEC")     = (int)COSIM_REG_VEC;
  m.attr("REG_VECELEM") = (int)COSIM_REG_VECELEM;
  m.attr("REG_CSR")     = (int)COSIM_REG_CSR;
}
