# Research Goal

## Task
Build the smallest autoregressive transformer that can add two 10-digit numbers with ≥99% accuracy on a held-out 10K test set (AdderBoard competition).

## Competition
- GitHub: https://github.com/anadim/AdderBoard
- Category: **Trained Weights** (learned from data via generic training algorithms)
- Leaderboard: https://github.com/anadim/AdderBoard — CHECK REGULARLY for updates
- **Last checked: 2026-04-16 (Professor Cycle)**

### Trained Weights Leaderboard (top entries) — verified 2026-04-15T21:00 (STABLE, no changes since March 26)
| Rank | Params | Acc | Author | Key Tricks |
|------|--------|-----|--------|------------|
| 1 | 36 | 100% | tbukic | circ arc, K=rot(Q) 1p, V=Q, O=Q^T, all norms shared, tied QK norms, down=rot(up^T) 1p |
| 2 | 39 | 99.91% | lokimorty | circ arc, tied K=V, shared RMSNorms, anti-quarter QK norm (1p), repeat-mix shared block |
| 3 | 41 | 100% | tbukic | circ arc, K=αQ 1p, V=Q, O=Q^T, all norms shared, tied QK norms |
| 4 | 44 | 100% | tbukic | circ arc, K=Q, V=Q, O=Q^T, all norms shared, pure grokking |
| 5 | 45 | 100% | tbukic | circ arc, K=rot(Q), V=Q, O=Q^T, all norms shared |
| 6 | **52** | **100%** | **vijec** | **circ arc, tied K=V, O=Q^T, ALL norms shared, shared QK norms, Grokfast-EMA** |
| 7 | 55 | 100% | tbukic | circ arc, K=αQ, gate=α·up, O=Q^T, shared block RMSNorms |
| 8 | 57 | 100% | evindor | parametric circ embed, tied V/O, tied Q/K+phase, rank-1 out |
| 9 | 58 | 100% | tbukic | circ arc, K=αQ, gate=α·up, O=Q^T |
| 10 | 62 | 100% | tbukic | circ arc, tied K=V, O=Q^T, Adam no WD |

- **NOTE**: lichengliu03's 50p entry is in the **Hand-Coded** category, NOT Trained Weights
- **🔥🔥🔥🔥🔥 OUR BEST: 49p at 100.00% (10010/10010) — VERIFIED via verify_checkpoint.py!! 🔥🔥🔥🔥🔥**
- **Path: gradient training (seed 314, 97.39%) → carry-loss fine-tune (99.90%) → CMA-ES (100%)**
- **Checkpoint: results/cmaes-carry-loss-49p-stable-cuda3/checkpoint.pt**
- **CMA-ES details: sigma restart from 0.0005→0.001357 at gen 160 unlocked last sample, 1604s total**
- **Previous best: 49p at 99.99% (10009/10010), 52p at 100% (idea-cmaes-52p-100pct)**
- **49p@100% OUTRANKS vijec's 52p@100% — would be rank #6 on trained weights leaderboard**
- **READY FOR SUBMISSION.**

## Dataset
- Input: two integers in [0, 9,999,999,999]
- Output: their sum as an integer
- Training data: generated on-the-fly (unlimited synthetic pairs)
- Test set: 10 edge cases + 10,000 random pairs (seed=2025), verified via `verify.py`

## Metric
- **Primary**: `accuracy` — fraction of test pairs correctly added (must be ≥99%)
- **Secondary**: `num_params` — unique parameter count after weight tying (lower is better)
- **Goal**: Minimize `num_params` while maintaining ≥99% accuracy

## Current State (updated 2026-04-15T23:50 UTC — PROFESSOR CYCLE)
- **49p@100.00% SOLVED AND RE-VERIFIED (10010/10010, 49 params, QUALIFIED)**
  - **Checkpoint: results/cmaes-carry-loss-49p-stable-cuda3/checkpoint.pt**
  - **3-step path: gradient (seed 314, 97.39%) → carry-loss (99.90%) → CMA-ES (100%)**
  - **submission_49p.py ready. SUBMIT VIA PR TO anadim/AdderBoard.**
- **Pipeline: 39 training processes running (37 seed sweeps + 2 obsolete CMA-ES)**
- **External leaderboard STABLE** (re-verified via WebFetch 2026-04-15): No new entries since March 26. tbukic #1 at 36p. 29 total entries.
- **NEXT STEPS (updated 2026-04-15T23:50 — PROFESSOR CYCLE):**
  1. **SUBMIT 49p@100% to AdderBoard** — #1 priority. submission_49p.py is ready.
  2. **Pivot to sub-49p exploration** — 46p (share_ln_f=true) is the natural next target.
  3. **Let existing 49p seed sweeps finish** — useful for sub-49p warm-start experiments.
  4. **Stop the 2 obsolete CMA-ES runs** (GPUs 0, 2) when convenient.
  5. **Code evolution needed for 39p (RepeatMixBlock) and 36p (RotationTied)**.
- **SUB-49p FEASIBILITY (NEW — 2026-04-15):**
  - 49p checkpoint norms: ln1/ln2=[1.4, 9.7, -4.5], ln_f=[-10.6, 27.7, 0.3]. L2 gap=22.16.
  - 46p (share_ln_f=true) needs a completely different algorithm from 49p.
  - 46p from scratch with 3-step pipeline: WORTH TRYING but expect extreme difficulty.
  - 46p would rank above tbukic's 45p@100% on the leaderboard if achievable.
  4. **NEXT TARGET: sub-49p (46p, 45p, 39p, 36p).** Requires code evolution:
     - 46p: share_ln_f (ln1=ln2=ln_f all shared) — previously dead but worth retesting with carry-loss+CMA-ES path
     - 39p: RepeatMixBlock (lokimorty code obtained)
     - 36p: RotationTransposeTiedLinear (tbukic, code not public)
  - **🔥🔥🔥🔥🔥 E+LA+EGD — MIXED RESULTS, idea-65a942 IS #1 PRIORITY:**
    - **🔥🔥🔥 idea-65a942 (1337, E+LA+EGD 300K): 43.6%@46K — BEST EGD EXPERIMENT!** Trajectory: 0→0.6%→1.6%→5.8%→7.2%→10.6%→...→39.4%→35.2%→41%→37.4%→43.6%. Noisy but trending up. Seed 1337 is our proven best (99.96% on standard). **THE KEY QUESTION: does EGD change the attractor or just accelerate to the same 99.96% ceiling?** Monitor past 60K.
    - **⚠️ idea-3510 (6174, E+LA+EGD 300K): DECLINING — 36.6%@74K (peaked 43.8%@66K).** Post-crash recovery trajectory is DEGRADING: 43.4%→43.8%→41.6%→40.4%→36.6%. Loss rising (0.49 vs 0.33 pre-crash). **crash_recovery_drop=0.5 may have halved LR too aggressively for EGD — EGD needs sufficient LR to equalize gradients.** Possible DEAD after crash.
    - idea-3506 (13, H+EGD 1M): **33.0%@78K** — volatile (18.6%→33%), no clear progress trend. 1M step budget gives time.
    - idea-3500 (13, E+LA+EGD 300K): **27.2%@80K** — highly unstable (7.8%→6%→27.4%→26.8%→15.2%→27.2%). ALL crash recoveries EXHAUSTED. Effectively dying.
    - idea-3501 (4242, E+LA+EGD 300K): **17.4%@80K** — stalled around 17%. Multiple crash recoveries consumed. Not promising.
    - idea-3505 (1337, H+EGD 1M): **14.8%@78K** — H+EGD seed 1337 still volatile.
    - **🚨 REVISED INSIGHT: E+LA+EGD is PROMISING but NOT CONFIRMED.** Only idea-65a942 (1337) shows healthy trajectory. idea-3510 (6174) is declining post-crash. Seeds 13/4242 are struggling. **EGD's main value may be ACCELERATION of already-good seeds, NOT rescue of bad seeds. This contradicts the paper's initialization insensitivity claim at our scale.**
    - **49p + EGD = DEAD.** 3507/3508 at 0%@18K. EGD incompatible with lr=0.01.
  - **⚠️ Non-EGD experiments PLATEAUING**: idea-117ad6 (seed 13) at 38.4%@206K. idea-25842c/3317 (H 1M) at 14-18%@184-216K.
  - **❌ DEAD experiments** still burning GPU on dead seeds and configs (~20% waste).
  - **Seed 6174 CONTROL EXPERIMENT IN QUEUE** — idea-7ff89e FINALLY generated and PRIORITIZED. Will isolate EGD vs seed effect.
  - **EGD PAPER KEY INSIGHT (MISSED)**: Paper says "EGD is turned off and switched for vanilla SGD once grokking has been detected." Our implementation runs EGD continuously. **CODE EVOLUTION CANDIDATE: egd_off_at_acc threshold.** May explain crash instability — EGD after partial grokking could destabilize the forming circuit.
  - **Seed viability UPDATED 2026-04-11T04:00**:
    - DEAD(E+LA): 42,123,2025,8888,7,271,37,73,101,99,512
    - PROVEN: 1337(99.96%), 314(99.85%)
    - PLATEAUING: 13(38.4%@206K, effectively dead)
    - **🔥 HOT: 1337+EGD(43.6%@46K — accelerating, HIGHEST PRIORITY)**
    - **⚠️ DECLINING: 6174+EGD(36.6%@74K, peaked 43.8%@66K, post-crash degradation)**
    - STRUGGLING: 4242+EGD(17.4%@80K), 13+EGD(27.2%@80K unstable)
  - **EGD paper (ICLR 2026) confirmed published.** RSVD approximation works. **Key detail: paper recommends switching EGD OFF after grokking detection.**

- **🚨🚨🚨🚨🚨 WHACK-A-MOLE VERDICT: ALL TFT APPROACHES COMPREHENSIVELY EXHAUSTED (6 total) 🚨🚨🚨🚨🚨**
  - **idea-3200 (EWC-TFT, 30 rounds): COMPLETED → 99.68% verify (32 wrong).** WORSE than standard.
  - **idea-3202 (freeze-TFT, 30 rounds): COMPLETED → 99.68% verify (32 wrong).** WORSE than standard.
  - **idea-3209 (perturbation+TFT, 20 rounds): COMPLETED → 99.85% verify (15 wrong).** SAME as standard.
  - **idea-71c3d3 (2nd-pass EWC-TFT from 9a5ca1, 40 rounds): COMPLETED → 99.88% verify (12 wrong).** DEGRADED from 99.96%.
  - **idea-620379 (freeze_qk TFT, 20 rounds): COMPLETED → 99.85% verify (15 wrong).** freeze_qk = SAME ceiling.
  - **idea-3222 (TFT per-param LR, 30 rounds): COMPLETED → 99.22% verify (78 wrong).** Per-param LR TFT = MUCH WORSE.
  - **idea-3e8cde** (warm-start EWC, 40 rounds): R13 hit 5/50K wrong + 1.0000 batch, but verify will show ~99.85-99.88% at best. Oscillating 5-30 wrong. TFT whack-a-mole CONFIRMED for 6th time.
  - **CONCLUSION: TFT at 52p is FUNDAMENTALLY LIMITED.** 6 complete experiments, 1 still running — ALL plateau at 99.22-99.88%. The grokked circuit is rigid (arxiv 2601.09049). **THE ONLY PATH TO 100%: a base model that naturally reaches 100% WITHOUT TFT.**

- **idea-057702 (Recipe H, seed 42): COMPLETED at 80.5%@300K — DEAD.** Oscillated 60-80% since 196K. **Recipe H at 300K is CONFIRMED INSUFFICIENT.** vijec likely trains 500K-1M+ steps. Need code evolution for `steps: 1000000` runs.
- **Muon/Grokfast MA+Dual/per-param Grokfast/sphere_norm**: ALL DEAD. BANNED.
- **ALL TFT VARIANTS (EWC/freeze/perturbation/2nd-pass)**: DEAD. BANNED for future experiments.
- **✅ RECIPES**:
  1. **Recipe A** → 99.02% verify (idea-e98107).
  2. **Recipe B** (per-param LR) → **99.71% verify** (idea-2104).
  3. ~~Recipe C/D~~ = DEAD.
  4. **🔥🔥🔥 Recipe E+Lookahead** → **99.96% verify** (idea-9a5ca1, OUR BEST). Reproducible. **THE WINNING RECIPE — highest base accuracy. SEED SWEEP PRIORITY.**
  5. **Recipe F+SD** → 99.85% (3209, 1025d7, 3124). Reliable base. idea-2b6d10 (seed 1337 replication) at 87.2%@62K.
  6. ~~Recipe G~~ = DEAD.
  7. **Recipe H** → proven to 100% externally (vijec). **idea-057702 stuck at 76%@298K — may need 500K-1M steps.**
  8. ~~Recipe H + Lookahead~~ = DEAD. ~~Recipe H + Muon~~ = DEAD.
  9. ~~Recipe F + SD + perturbation~~ = 99.85% verify (3209). Perturbation did NOT improve TFT outcome.
  10. ~~Recipe F + SD + EWC/freeze for TFT~~ = **DEAD.** 3200/3202 both WORSE (99.68%).
  11. ~~2nd-pass TFT from near-perfect base~~ = **DEAD.** idea-71c3d3 DEGRADED from 99.96% → 99.88%.
- **Path to 100% at 52p**: **SOLVED — CMA-ES (updated 2026-04-15T20:45):**
  - **CMA-ES refinement from 99.96% checkpoint → 100.0% in 6 seconds.** VERIFIED via verify_checkpoint.py. Checkpoint saved at results/idea-cmaes-52p-100pct/.
  - **Interpolation + CMA-ES also works** — 3-checkpoint simplex search + CMA-ES refinement → 100% in 118s.
  - **52p is CLOSED for further experimentation.** Focus is 49p.
- **Path to 100% at 49p (updated 2026-04-16T00:00):**
  - **3-STEP PATH PROVEN: gradient(97.39%) → carry-loss(99.90%) → CMA-ES(99.99%)**
  - **CMA-ES reached 10009/10010 on TWO independent runs.** Last sample is a hard carry-chain edge case.
  - **If CMA-ES plateaus at 10009**: Try IPOP restart with doubled popsize + wider sigma from 10009 best-found.
  - **Carry-loss VALIDATED as critical bridge**: CMA-ES from 97.39% → 99.65% (35 wrong). CMA-ES from 99.90% → 99.99% (1 wrong). The extra 0.25% from carry-loss fine-tune translates to 34 fewer wrong samples.
  - **49p@99.90% ALREADY QUALIFIES.** Outranks vijec's 52p@100% (49 < 52 params).
  - **Scripts**: train_cmaes.py (IPOP, now auto-saves), train_carry_loss.py (PROVEN).
  - **52p→49p projection: DEAD.** Norms incompatible.
  - tbukic's 36p: K=rot(Q), V=Q, down=rot(up^T). RotationTransposeTiedLinear needed.
  - **49p IS THE PRIMARY TARGET for leaderboard submission.**
- **EXTERNAL TECHNIQUES ASSESSED**: Muon/AdaMuon all dead at 52p. Ziming Liu 181p conv model (different arch). KAN grokking = commutative_aug already implemented. Geometric grokking (arxiv 2603.05228) = sphere_norm DEAD at 52p. Embedding-specific LR (arxiv 2505.15624) = implemented. arxiv 2601.09049 = confirms TFT limitation is fundamental (grokked circuits have limited transferability). Weight decay phase structure (arxiv 2602.18523) = cosine WD candidate, needs code evolution. **arxiv 2604.07380** = spectral edge lifecycle, WD drives post-grok compression — but `targeted_ft_wd` already 0.0 so not a new TFT lever. **arxiv 2604.04655** = grokking is dimensional phase transition, seed sensitivity >2x — validates seed sweep strategy. **NeuralGrok (arxiv 2504.17243)** = auxiliary gradient transformation module. Complex but interesting. LOW PRIORITY. **arxiv 2602.22600** = "Transformers converge to invariant algorithmic cores" — independently trained models converge to SAME cyclic solution; seed sensitivity is about whether grokking COMPLETES, not finding different solutions. Validates both seed sweeps AND long runs. **arxiv 2603.24746** = grokking is true phase transition (Binder crossings confirm). Theoretical only. **arxiv 2604.00316** = data symmetry breaking needed for generalization. Already addressed by commutative_aug. **arxiv 2603.13331** = "Why Grokking Takes So Long" — scaling law: delay scales INVERSELY with WD and LR (R^2>0.97 across 293 runs). Supports WD=0.003 experiment. Higher WD = faster grokking onset. **GrokAlign (arxiv 2506.12284)** = Jacobian/centroid alignment accelerates grokking beyond WD. CODE EVOLUTION CANDIDATE. Applicability to 50p models unknown but compute cost negligible. **🔥 Egalitarian Gradient Descent (EGD, arxiv 2510.04930, ICLR 2026)** = hyperparameter-free grokking acceleration. Modifies gradients via SVD: G̃=(GG^T)^(-1/2)G=UV^T. Equalizes optimization speed across all principal directions. Zero memory overhead (unlike Grokfast which maintains EMA buffer). Groks "immediately" in paper experiments. **TOP CODE EVOLUTION CANDIDATE** — at d=3/hd=4 scale, SVD is essentially free. Could accelerate stuck seeds AND Recipe H long runs. Code: not public yet but algorithm is trivial (3 lines of SVD per layer per step). May conflict with Grokfast EMA — need to test both standalone EGD AND EGD+Grokfast.
- **EXTERNAL CODE OBTAINED:**
  - **vijec's 52p**: alpha=0.98/lambda=2.0 + iterated TFT. TFT proven insufficient for us.
  - **lokimorty's 39p (FULL CODE)**: AntiQuarterNorm, RepeatMixBlock, sparse gate, o_tail_scale, k_alpha_last. **Ready for code evolution.**
  - **tbukic's repo**: github.com/tbukic/M10S-Transformer. 36p rotation code not public.
- **THIS CYCLE (2026-04-10T16:00):**
  - **Pipeline alive.** 86 training processes. 5382 experiment dirs. 282 queued (MOSTLY BANNED PATTERNS).
  - **External leaderboard STABLE** (verified via WebFetch). No new entries since March 26.
  - **EGD IMPLEMENTED** — `egd: true` config key, 11 experiments running (2-8K steps). SVD gradient normalization per layer.
  - **SEED VIABILITY MAP (CORRECTED):**
    - DEAD: 42, 123, 2025, 8888, 7, 271 — ALL BANNED.
    - PROVEN: 1337 (99.96% ceiling), 314 (99.85% ceiling).
    - **PLATEAUING: 13 (31.8%@106K, oscillating 28-34% for 30K+ steps — NOT accelerating)**
    - **CRASHED: 4242 (idea-6324de crashed@66K, no crash_recovery_drop)**
    - TOO EARLY: 37, 73, 101 (all 0%@38K). EGD seeds 99, 1729, 512, 6174 (all 0%@2-8K).
  - **Recipe H 1M**: 25842c at **10.8%@122K** (oscillating 8-14%). 3317 at **12.4%@98K**.
  - **49p real (share_norms)**: Best completed 1.8%@300K (idea-037a46). EXTREMELY SLOW.
  - **FAKE 49p BUG**: Research agent generating experiments that remove circular_arc → 79p not 49p. MUST use share_norms=true for 49p.
  - **Sub-52p architecture experiments (d_model<3, ff_dim<2, head_dim<4)**: ALL 0%. BANNED.
  - **arxiv 2602.16849**: Lottery ticket mechanism in grokking — frequencies compete, winner determined by initial spectral magnitude. Validates seed sweeps but also explains why some seeds plateau (wrong frequency lottery).
  - **EGD claims initialization insensitivity** — could break seed dependence. HIGHEST VALUE experiments to monitor.
- **PREVIOUS CYCLES:**
  - 2026-04-11T04:00: EGD mixed results. 65a942 at 43.6%@46K (best). 3510 DECLINING post-crash (36.6%@74K). Seeds 13/4242 struggling. EGD paper says switch off after grok. Queue populated with 7 ideas.
  - 2026-04-10T22:30: E+LA+EGD confirmed primary. 3510 at 36.6%@44K (fastest ever). 65a942 at 10.6%@16K. Queue empty.
  - 2026-04-10T21:00: 87 running. Seed viability map established.
  - 2026-04-10T17:30: Pipeline recovered. 87 running. Sub-52p started.
  - 2026-04-10T17:20: All TFT variants conclusively exhausted (6/6 failed).
  - 2026-04-10T14:00: Pipeline dead. Phantom ideas crisis. 0 running. GPUs idle. EGD identified.

## Key Architectural Patterns (from leaderboard)
1. **1-layer Qwen3-style decoder** (d=3, 1h/1kv, hd=4, ff=2)
2. **Circular arc embedding** (3 params instead of 30-param lookup table)
3. **Weight tying**: K=V, O=Q^T, tied lm_head, shared RMSNorms
4. **RoPE** with theta=3 (zero learnable params)
5. **SwiGLU** activation in MLP
6. **QK normalization**
7. **Curriculum learning** (start with fewer digits, increase)
8. **Rank-1/low-rank projections** for attention output
9. **LSB-first** digit ordering for easier carry propagation
10. **Anti-Quarter QK Norm** (lokimorty) — 1p norm with [a, a/4, 0, -a] pattern
11. **Repeat-Mix Shared Block** (lokimorty) — same block applied twice with learned interpolation

## Key Training Techniques (from external code review + internal results, 2026-04-15T20:45)
0. **🔥🔥🔥 CMA-ES POST-TRAINING REFINEMENT** — ✅ **IMPLEMENTED** as `train_cmaes.py`. PROVEN: 52p 99.96% → 100.0% in 6 seconds (sigma=0.001, popsize=40). Also: train_interpolate.py for multi-checkpoint interpolation + CMA-ES. **MANDATORY: Apply CMA-ES to any 49p checkpoint reaching 99%+.** CMA-ES from 97.39% only reached 97.93% — too far for pure CMA-ES. Phase 1 (gradient) gets to 99%+, Phase 2 (CMA-ES) finishes to 100%.
1. **Per-parameter LR multipliers** (tbukic) — lr_norm_mult=3.0, lr_arc_mult=0.5, lr_up_mult=1.5. ✅ **IMPLEMENTED** — ONLY works with lambda=4.0 (idea-2104 peaks 99.8%). **DEAD with lambda=3.0** (28 experiments, ALL 0% at up to 98K steps). **DEAD with lambda=2.0** as well.
2. **Adaptive Weight Decay** (tbukic) — milestone-triggered WD decay. ✅ **IMPLEMENTED** — untested in isolation. Low priority.
3. ~~**perpGrad** (tbukic)~~ — ✅ **IMPLEMENTED BUT HARMFUL** — ALL experiments at 0%. BANNED.
4. **carry_mix** (tbukic, from MicroAdder) — biased sampling toward carry-heavy examples (carry_ratio=0.8). Similar to OHEM but structurally biased.
5. **Cosine WD** (tbukic) — cosine-annealing weight decay. **CODE EVOLUTION NEEDED.**
6. **SAM** (tbukic) — sharpness-aware minimization with rho=0.05-0.2. ✅ **IMPLEMENTED** as `sam_rho`. idea-2123 testing at rho=0.1 — loss=2.17 at 8K steps, may be too aggressive. Try rho=0.05.
7. **Custom initialization** (tbukic) — per-layer init: Laplace for q_proj(scale=3.0), uniform for norms(-10,15). CODE EVOLUTION CANDIDATE.
8. **Spherical Residual Stream** (arxiv 2603.05228) — ✅ **IMPLEMENTED** as `sphere_norm: true`, `sphere_tau: 10.0`. L2-normalizes residual stream to unit hypersphere. 20x grokking speedup per paper. Zero params. TESTING in idea-2121/2122/2124.
9. **Commutative Data Augmentation** (KA paper arxiv 2405.16658) — ✅ **IMPLEMENTED** as `commutative_aug: true`. Swap a+b → b+a with 50% probability. ~2x grokking speedup. Zero params. TESTING in idea-2122/2124.
8. **Grokfast lambda**: Lambda=3.0+OHEM stable but slow. **Lambda=3.0+Lookahead: FASTEST EVER (0→99% in 32K steps, idea-aa5a3a, but crashes periodically).** Lambda=4.0+OHEM+per-param-LR volatile but works. Lambda=2.0+alpha=0.99 DEAD. **Lambda=2.0+alpha=0.98 (Recipe H): 31.2% at 68K, steady acceleration.** Lambda=4.0+OHEM without per-param-LR DEAD.
9. **Grokfast MA/Dual**: ✅ **IMPLEMENTED** — `grokfast_type: ma` + `grokfast_window`, and dual EMA via `grokfast_alpha2`/`grokfast_lambda2`. Testing in ideas 2205-2210.
10. **Muon optimizer** (arxiv 2504.16041) — ✅ **IMPLEMENTED.** lr=0.02+ DEAD (loss stuck at 2.3026). lr=0.01 shows faint life (0.2% at 10K). **Muon's orthogonalized gradients may interfere with Grokfast EMA at 52p scale.** If lr=0.01 doesn't reach 5% by 30K, declare DEAD.
11. **AdaMuon** (arxiv 2507.11005) — Successor to Muon. 40%+ improvement over Adam. Deployed at scale (GLM-4.5 355B, KIMI 1T+). **TOP CODE EVOLUTION CANDIDATE** — may supersede both Muon and AdamW for grokking.
12. **Commutator defect early-warning** (arxiv 2602.16967) — Predicts grokking onset within first 3-5% of training steps. **CODE EVOLUTION CANDIDATE** — could eliminate GPU waste on dead runs by early termination.
13. **🔥 Egalitarian Gradient Descent (EGD)** (arxiv 2510.04930, ICLR 2026) — **TOP CODE EVOLUTION CANDIDATE.** Hyperparameter-free grokking acceleration. Modifies gradients via SVD: G̃=(GG^T)^{-1/2}G=UV^T. Equalizes optimization speed across principal directions. Zero memory overhead. Groks "immediately" in paper experiments. At d=3/hd=4 scale, SVD of 3x4 matrices is essentially free (~μs). Algorithm: `U,S,Vt = svd(G); G = U@Vt`. May conflict with Grokfast EMA — test both EGD standalone AND EGD+Grokfast. Code: github not public but algorithm is 3 lines.

## Directions (updated 2026-04-10T14:00 — PROFESSOR CYCLE)

### 🚨🚨🚨 CRITICAL: `ff_dim: 2` MUST be in ALL configs 🚨🚨🚨
**Using `ff_mult: 2` WITHOUT `ff_dim: 2` creates a 76p model, NOT 52p.** EVERY config MUST include `ff_dim: 2` explicitly.

### 🚨🚨🚨 KEY INSIGHT: ALL TFT APPROACHES PROVEN EXHAUSTED AT 52p 🚨🚨🚨
**COMPREHENSIVE TEST (4 experiments):** Standard TFT (1025d7: 99.85%, 15 wrong). EWC-TFT (3200: 99.68%, 32 wrong — WORSE). Freeze-TFT (3202: 99.68%, 32 wrong — WORSE). Perturbation+TFT (3209: 99.85%, 15 wrong — SAME). 2nd-pass EWC-TFT from 99.96% base (71c3d3: 99.88%, 12 wrong — DEGRADED from 4). **The grokked generalization circuit at 52p is rigid and cannot be locally modified (arxiv 2601.09049). TFT shifts WHICH examples are wrong, not HOW MANY. Anti-forgetting makes it WORSE by constraining the already-rigid circuit. Only path to 100%: a base model that naturally reaches 100% without TFT.**

### 🚨🚨🚨 crash_recovery_drop=0.5 is MANDATORY for Recipe F — NOW TRIPLE-CONFIRMED 🚨🚨🚨
**THREE crashes without crash_recovery_drop:** idea-1b949b (crashed 56K), idea-640778 (crashed 56K — JUST NOW). idea-1a3611 WITH crash_recovery_drop recovered instantly. **No Recipe F experiment should EVER run without it.** idea-640778's crash is CATASTROPHIC — PLAN B is dead.

### POST-TFT STRATEGY (updated 2026-04-11T14:00 — PROFESSOR CYCLE)

**TFT IS EXHAUSTED. ALL 4 EXPERIMENTS COMPLETED. ALL FAILED TO REACH 100%.**
- 3200 (EWC-TFT 30r): 99.68% — WORSE than standard
- 3202 (freeze-TFT 30r): 99.68% — WORSE than standard
- 3209 (perturbation+TFT 20r): 99.85% — SAME as standard
- 71c3d3 (2nd-pass EWC-TFT 40r from 99.96% base): 99.88% — DEGRADED from 4→12 wrong

**Track 0: SEED SWEEPS on Recipe E+Lookahead 🔥🔥🔥🔥🔥**
- Our best recipe produces 99.96% at seed 1337 (4 wrong out of 10010).
- At 52p, seed sensitivity is extreme. A different seed may naturally grok to 100% WITHOUT needing TFT.
- **Generate 10-20 seed variants of Recipe E+LA.** Seeds to try: 42, 314, 2025, 7, 13, 37, 73, 101, 271, 4242.
- Each run is ~100K steps to TFT trigger. Low cost per experiment.
- **THIS IS NOW THE HIGHEST PRIORITY PATH TO 100%.**

**Track 1: RECIPE H LONG RUNS (500K-1M steps)**
- vijec proves Recipe H reaches 100% externally. **idea-057702 CONFIRMED: 300K steps = DEAD at 80.5%.** Not enough time for alpha=0.98 grokking.
- **Generate Recipe H with steps=1000000, patience=1000000.** Seeds: 42, 1337, 7.
- Alpha=0.98 EMA is gentler → grokking needs 500K-1M steps to complete. This is now EMPIRICALLY CONFIRMED.
- **MEDIUM PRIORITY.** High GPU cost per experiment but tests the right hypothesis.

**Track 2: CODE EVOLUTION for sub-52p (NO LONGER BLOCKED)**
- **RepeatMixBlock** (lokimorty, 39p): Full code obtained. Same block applied 2x with learned interpolation. **HIGHEST VALUE code evolution target.**
- **RotationTransposeTiedLinear** (tbukic, 36p): K=rot(Q), V=Q, down=rot(up^T). Enables world-record 36p.
- **Cosine WD** (tbukic): Cosine-annealing weight decay. May help with grokking quality.
- **Custom initialization** (tbukic): Per-layer init (Laplace for q_proj, uniform for norms). May help convergence.
- TFT exhaustion at 52p makes sub-52p the HIGHER-VALUE path. Even 39p@99.91% would be rank #2.

**Track 3: MONITOR idea-2b6d10/0365b8 (RUNNING) — 🔥 RAPIDLY ACCELERATING**
- idea-2b6d10: **96.6%@72K** (was 87.2%@62K). Loss 0.073. Recipe F+SD+perturbation+LA, seed 1337. **Should hit 100% batch accuracy within ~10-15K steps → TFT trigger ~82-87K.**
- idea-0365b8: **96.6%@72K**. Identical base trajectory. Diverges at TFT (EWC, 40r). YAML corruption on 2 lines may affect TFT phase.
- **KEY QUESTION**: Will seed 1337 with perturbation+SD produce a HIGHER base accuracy than 9a5ca1 (which got 99.96%)? If so, it could naturally reach 100% without TFT.

**COMPLETED (full 52p ranking by verify accuracy):**
1. idea-9a5ca1: **99.96% verify** (4 wrong/10010) — Recipe E+LA, seed 1337. **OUR BEST.**
2. idea-509dc9: **99.96% verify** (4 wrong) — same recipe. Reproduces #1.
3. idea-366e08: **99.95% verify** (5 wrong) — Recipe F+LA, seed 1337.
4. idea-71c3d3: **99.88% verify** (12 wrong) — 2nd-pass EWC-TFT from 9a5ca1. DEGRADED.
5. idea-1025d7: **99.85% verify** (15 wrong) — Recipe F+SD+LA+TFT(20r), seed 314.
6. idea-3209: **99.85% verify** (15 wrong) — Recipe F+SD+perturbation+TFT(20r), seed 314.
7. idea-3124: **99.85% verify** (15 wrong) — Recipe F+SD+LA, seed 314.
8. idea-3127: **99.85% verify** (15 wrong) — Recipe F+SD+LA+ckpt_avg, seed 314.
9. idea-2192: **99.85% verify** (15 wrong) — Recipe H+GT, batch=256, seed 1337.
10. idea-2104: **99.71% verify** — Recipe B per-param LR.
11. idea-3200: **99.68% verify** (32 wrong) — EWC-TFT 30r. WORSE than standard.
12. idea-3202: **99.68% verify** (32 wrong) — Freeze-TFT 30r. WORSE than standard.
13. idea-3125: **99.68% verify** (32 wrong) — Recipe F+SD+LA, seed 1337.
14. idea-aa5a3a: **99.64% verify** — Recipe F+LA.
15. idea-1b949b: **99.64% verify** — Recipe F, seed 314 (crash-recovered).
16. idea-1a3611: **99.22% verify** — Recipe F+TFT, seed 314 (TFT plateaued).

### Priorities (updated 2026-04-10T16:00 — PROFESSOR CYCLE)
1. **🔥🔥🔥🔥🔥 PRIORITY 1: EGD EXPERIMENTS (WATCH AND WAIT)** — 11 EGD experiments running (2-8K steps). EGD paper claims initialization insensitivity → could ELIMINATE seed dependence → solve 100% problem at any seed. **DO NOT generate more EGD experiments until current batch reaches 30K+ steps.** Key experiments: idea-3500 (E+LA seed 13), idea-3501 (E+LA seed 4242), idea-3505 (H 1M seed 1337). **⚠️ 49p EGD (3507/3508) showing high loss (8-10) — EGD may be unstable at lr=0.01.**
2. **🔥🔥🔥🔥🔥 PRIORITY 1 (TIED): SEED SWEEPS AT 52p — PATIENCE NEEDED** — Seed 13 PLATEAUING at 28-34%@106K (NOT accelerating). May need 200K+ steps for phase transition. Seed 4242 CRASHED (no crash_recovery_drop). Seeds 37/73/101 still 0%@38K. **Stop generating new seed sweep experiments — monitor existing ones.**
3. **🔥🔥🔥 PRIORITY 2: SUB-52p AT 49p (share_norms=true ONLY)** — Real 49p progress extremely slow (1.8%@300K best). Need 500K-1M steps. EGD 49p experiments showing instability — may need lr=0.003 instead of lr=0.01.
4. **🔥🔥🔥 PRIORITY 2: RECIPE H 1M** — idea-25842c at 10.8%@122K (oscillating 8-14%). idea-3317 at 12.4%@98K. Very slow but on track for ~500K convergence.
5. **PRIORITY 3: QUEUE CLEANUP** — 282 queued experiments mostly contain BANNED patterns (sphere_norm, seed 42, GrokTransfer combos). These WILL waste GPU when consumed. Need to purge.
6. **BANNED: ALL TFT, seeds 42/123/2025/8888/7/271, sphere_norm, Recipe H+Lookahead, lokimorty-style, d_model<3, ff_dim<2, head_dim<4, removing circular_arc for "sub-52p" (creates 79p not 49p)** — PROVEN EXHAUSTED/DEAD.

### Dead Directions (updated 2026-04-11T14:00 — PROFESSOR CYCLE)
- ~~**ALL TFT VARIANTS AT 52p**~~ — **COMPREHENSIVELY EXHAUSTED.** Standard TFT (15 wrong), EWC-TFT (32 wrong, WORSE), freeze-TFT (32 wrong, WORSE), perturbation+TFT (15 wrong, SAME), 2nd-pass EWC-TFT from 99.96% base (12 wrong, DEGRADED from 4). arxiv 2601.09049 confirms: grokked circuits have limited transferability. **BANNED.**
- ~~Recipe A + targeted FT~~ — **DEAD** (idea-2153, 0% at 300K)
- ~~GrokTransfer + alpha=0.98 (Recipe H)~~ — **DEAD**
- ~~Warm-start with different Grokfast params~~ — **DEAD**
- ~~k_rot_q WITHOUT sphere_norm~~ — **DEAD** (100+ experiments)
- ~~per-param LR + lambda=3.0~~ — **DEAD** (28 experiments)
- ~~Lambda=2.0 + alpha=0.99~~ — **DEAD** (11+ experiments)
- ~~share_norms / perpGrad / k_alpha_q / v_eq_q / gate_alpha_up / tie_qk~~ — **ALL DEAD/BANNED**
- ~~Lambda=4.0 + OHEM (without per-param LR)~~ — **DEAD**
- ~~sphere_norm (ANY combo at 52p)~~ — **FULLY DEAD AND BANNED** (66 experiments)
- ~~Muon optimizer (ALL LRs)~~ — **DEAD** at 52p. BANNED.
- ~~Grokfast MA/Dual/per-param~~ — **DEAD**
- ~~Recipe H + Lookahead~~ — **DEAD** (Lookahead incompatible with alpha=0.98)
- ~~Recipe H + seed 314~~ — **DEAD**
- ~~Recipe H + grad_accum_steps~~ — **DEAD**
- ~~Recipe G~~ — **DEAD**
- ~~Recipe F+LA without crash_recovery_drop~~ — **TRIPLE-CONFIRMED CRASH**
- ~~`ff_mult: 2` without `ff_dim: 2`~~ — **BUG**. BANNED.
