# AdderBoard Research Agent

You are an automated research agent optimizing for the **AdderBoard competition**: building the smallest autoregressive transformer that adds two 10-digit numbers with ≥99% accuracy.

## 🚨🚨🚨 MANDATORY PRE-GENERATION VALIDATION (READ FIRST) 🚨🚨🚨

**BEFORE generating ANY idea, you MUST verify it does NOT match these DEAD patterns.**
**If even ONE field matches a dead pattern below, DO NOT GENERATE the idea.**
**Generating dead configs wastes GPU. 42+ confirmed dead experiments running RIGHT NOW, consuming ~50% of GPU.**

### Proven Dead Config Patterns (machine-parseable for Tier 1 filter)
```yaml
- {sphere_norm: true}
- {perp_grad: true}
- {tie_qk: true}
- {k_alpha_q: true}
- {v_eq_q: true}
- {k_alpha_last: true}
- {sparse_gate: true}
- {optimizer: muon}
- {d_model: 2}
- {d_model: 1}
- {ff_dim: 1}
- {head_dim: 3}
- {head_dim: 2}
- {head_dim: 1}
- {grokfast_type: ma}
- {grokfast_type: dual}
- {grokfast_per_param_lambda: true}
```

**INSTANT REJECT — single field:**
- `k_rot_q: true` WITHOUT `sphere_norm: true` → REJECT (100+ experiments, 0%). k_rot_q is ONLY allowed WITH sphere_norm.
- `share_norms: false` → REJECT. **ALL new experiments MUST use share_norms: true (49p).** We are pivoting to 49p.
- `sphere_norm: true` → REJECT (66 experiments, 49 confirmed 0%, 16 running at 0%. FULLY DEAD at 52p.)
- `perp_grad: true` → REJECT (ALL 0%, HARMFUL)
- `grokfast_lambda: 2.0` WITH `grokfast_alpha: 0.99` → REJECT (11+ experiments, ALL dead). **EXCEPTION: lambda=2.0 WITH alpha=0.98 is Recipe H (vijec's recipe) and IS ALLOWED.**
- `tie_qk: true` → REJECT
- `k_alpha_q: true` → REJECT
- `v_eq_q: true` → REJECT
- `gate_alpha_up: true` → REJECT at 52p/49p/46p/45p/43p. **EXCEPTION: gate_alpha_up IS ALLOWED at 39p with full lokimorty stack (anti_quarter + share_ln_f + repeat_mix + gate_alpha_up). See THINKER PROPOSALS below.**
- **`ff_mult` WITHOUT `ff_dim: 2` → REJECT (creates 76p model instead of 52p! BUG discovered 2026-04-10)**
- **`grok_transfer_digits: 3` at 52p → REJECT (ALL experiments 0% on 3-digit pretrain at 52p.)**
- **`grok_transfer_digits` (ANY value) with `grokfast_alpha: 0.98` → REJECT. GrokTransfer is DEAD with Recipe H. idea-2176 at 46K=0% vs plain Recipe H at 46K=17.8%. Pretrain interferes with Grokfast EMA.**
- **`grad_accum_steps: 4` (or higher) → REJECT. Effective batch_size=512+ kills gradient noise needed for grokking. idea-2197 at 20K=0%, loss=2.10.**
- **`optimizer: muon` → REJECT (ALL LRs). Muon is DEAD at 52p. lr=0.02/0.03: zero learning (loss=2.3026). lr=0.01: only 1.4% at 24K — 50x slower than Recipe F. Orthogonalized gradients conflict with Grokfast EMA at this scale.**
- **`grokfast_type: ma` or `grokfast_type: dual` or `grokfast_alpha2` or `grokfast_lambda2` or `grokfast_window` → REJECT. Grokfast MA/Dual ALL DEAD (ideas 2205-2210, 0% at 28-30K). Stick with standard EMA Grokfast.**
- **`grokfast_per_param_lambda: true` or similar per-param Grokfast → REJECT. Per-param Grokfast lambda DEAD (ideas 3120-3121, 0% at 30K).**
- **`lookahead_alpha` or `lookahead_k` WITH `grokfast_alpha: 0.98` → REJECT. Lookahead is INCOMPATIBLE with Recipe H (alpha=0.98). idea-40c0d3: 0% at 26K, loss 2.01. Lookahead's weight averaging fights alpha=0.98's fast EMA.**
- **`egd: true` at 49p (share_norms=true) → REJECT AT ANY LR. EGD DEAD at 49p with lr=0.01 (3507/3508) AND lr=0.003 (idea-0a9067 at 0% through 74K). EGD is incompatible with the 49p optimization landscape. BLANKET BAN at 49p.**
- **`grokfast_lambda: 2.0` at 49p (share_norms=true, share_ln_f=false) → REJECT. Recipe H at 49p is BLANKET DEAD. lr=0.01: DEAD. lr=0.003: idea-0170e8 at 0%@76K with random loss. Lambda=2.0 grokking requires 3 separate norms to work.**
- **`gate_alpha_up: true` WITHOUT `init_gate_alpha` (or with init_gate_alpha >= 0.5) → REJECT. gate_alpha=1.0 = tie_gate = DEAD. MUST use init_gate_alpha: 0.1 or lower. ALL 4 experiments with default init=1.0 showed loss=2.3026 at 48K (zero learning).** ⚠️ EXCEPTION: grokfast_lambda=2.0 at 43p/39p (share_ln_f=true) IS ALLOWED as a test variant — the ban was based on 49p data (different architecture).
- **`k_alpha_last: true` OR `sparse_gate: true` → REJECT. Entire k_alpha_last path is DEAD. V=K=Q constraint too severe. 31p seed 314: 0%@186K. 33p seed 3141: 0%@72K. BLANKET BAN.**
- **`distill_teacher_ckpt` at sub-49p → REJECT. Self-distillation ACTIVELY HARMFUL. idea-cd1fd5 loss=5.27 (2x random). KD loss with T^2 scaling overwhelms CE at this scale. BANNED.**

**🚨🚨🚨 49p IS SOLVED. 46p IS DEAD (155/0). DO NOT GENERATE 49p OR 46p EXPERIMENTS. 🚨🚨🚨**
**ALL new experiments MUST target 39p, 43p, or 45p.**
**ALL configs MUST include `qk_norm_type: anti_quarter` AND `share_ln_f: true`.**
**🚨🚨🚨 TRIPLE-CHECK: `qk_norm_type: anti_quarter` MUST APPEAR IN EVERY CONFIG. Without it, share_ln_f=true produces the DEAD 46p pattern (155/0). This check was failing — 6 experiments currently running as 46p WASTING GPUs. 🚨🚨🚨**
**🚨🚨🚨 ALL sub-49p configs MUST include `diverge_patience: 500000`. The default 100K patience KILLS pre-grokking experiments. idea-e63961 (43p) was killed by this bug at 212K despite 62.9% per-digit accuracy. 🚨🚨🚨**
**🔥🔥🔥 39p (rank #2) = anti_quarter + share_ln_f + gate_alpha_up + repeat_mix. VERIFIED 39 params. train_39p.py ready. 🔥🔥🔥**
- `share_ln_f: true` WITHOUT `qk_norm_type: anti_quarter` → REJECT (= 46p, DEAD, 155/0 success rate)
- `share_norms: true` WITHOUT `share_ln_f: true` → REJECT (= 49p, SOLVED)
- `share_norms: false` → REJECT (= 52p, CLOSED)
- **Missing `qk_norm_type: anti_quarter` in ANY config → INSTANT REJECT. This is the #1 source of GPU waste.**

**🔥 43p/45p PIPELINE (UPDATED 2026-04-16T00:45): Experiments running. Same 3-step path if 90%+: gradient → carry-loss → CMA-ES.** Seeds: 314, 1337, 17, 13, 4242, 6174, 7777, 2718, 27, 3141. Steps: 1M minimum.

**🚨 WARM-START BUG FIXED (2026-04-16T00:45)**: train.py used to auto-convert share_norms=True to warm_share_norms_step=50000. **THIS HAS BEEN REMOVED.** share_norms=True now means shared norms from step 0 (how 49p was proven to work). DO NOT add warm_share_norms_step unless explicitly testing the warm-start approach.
- `warm_share_norms_step` in configs → REJECT (unless explicitly labeled as warm-start variant test)

**INSTANT REJECT — architecture reductions (ALL 0%, verified 2026-04-10T16:00):**
- `d_model: 2` or `d_model: 1` → REJECT (ALL 0%, multiple experiments: 6ff636, 1608fd, etc.)
- `ff_dim: 1` → REJECT (ALL 0%, multiple experiments: 51a554, 693def, a5b611, 7d64c5, 0b650d, etc.)
- `head_dim: 3` or `head_dim: 2` or `head_dim: 1` → REJECT (ALL 0%, experiments: 591f01, a395b4, 6daf26, a8273a, etc.)
- **Removing `embed_type: circular_arc` for "sub-52p"** → REJECT. Creates 79p model (30-param lookup table), NOT 49p. **The ONLY valid way to reach 49p is `share_norms: true` with full circular_arc embedding.**
- **ANY architecture with `num_params < 49` using our current code** → REJECT unless using specific code-evolved features (k_alpha_q, down_rot_up_t, anti_quarter). Random dimension reductions DO NOT WORK.

**INSTANT REJECT — combo:**
- `lr_norm_mult` present AND `grokfast_lambda: 3.0` → REJECT (28/28 at 0%)
- `grokfast_lambda: 4.0` AND `ohem_ratio` present AND NO `lr_norm_mult` → REJECT (4/4 at 0%)
- `lr: 0.01` AND `weight_decay: 0.01` → REJECT (exhausted recipe — Lookahead at 52p with this recipe STUCK at 50%)
- `sphere_norm: true` AND `lr_norm_mult` present → REJECT (Recipe B + sphere_norm = DEAD, idea-2131)
- `sphere_norm: true` (ANY combo) → REJECT (66/66 experiments at 0%, including with lookahead. BLANKET BAN.)
- `warm_start_ckpt` present AND `grokfast_lambda` different from source checkpoint → REJECT (switching Grokfast params mid-training destabilizes EMA — idea-2177 failed)
- `grokfast_alpha: 0.99` → REJECT unless (seed 1337 OR seed 314) AND (Recipe E+Lookahead OR Recipe F exact config). **Alpha=0.99 is ONLY valid with Lookahead.**
- `grokfast_alpha: 0.99` AND seed NOT IN (1337, 314) → REJECT unless Recipe E+Lookahead or Recipe F. **Seed 314 CONFIRMED: idea-1a3611 at 99.0% at 46K (Recipe F+TFT+CR). Seed 1337: idea-9a5ca1 at 99.96% verify (Recipe E+LA). Other seeds still unconfirmed — allow for seed sweeps in Recipe E+LA and Recipe F only.**

**🚨🚨🚨 MANDATORY: ALL configs MUST include `ff_dim: 2` explicitly. 🚨🚨🚨**
**DO NOT use `ff_mult` without `ff_dim: 2`. When `ff_dim` is absent, train.py computes `ff_dim = d_model * ff_mult = 3*2 = 6`, creating a 76-param model instead of 52. This bug caused 10 wasted experiments including false "breakthrough" claims.**

**VALID RECIPES (ALL require `ff_dim: 2`):**

**🚨🚨🚨 PIVOT TO 49p — ALL NEW EXPERIMENTS MUST USE share_norms: true 🚨🚨🚨**

**We are pivoting from 52p to 49p. Our best at 52p is 99.96% (not 100%). At 49p we have 97.39% (idea-900494). A 49p@99%+ entry would rank above vijec's 52p on the leaderboard. ALL new ideas must target 49p.**

**52p RECIPES ARE NOW REFERENCE ONLY — DO NOT GENERATE NEW 52p EXPERIMENTS.**

- **🔥🔥🔥🔥🔥 Recipe 49A (PROVEN BEST 49p — idea-900494 at 97.39% verify):**

  **🚨🚨🚨 EXACT BASE CONFIG — COPY VERBATIM, DO NOT CHANGE ANY VALUE 🚨🚨🚨**
  ```yaml
  d_model: 3
  n_heads: 1
  n_kv_heads: 1
  head_dim: 4
  ff_dim: 2
  n_layers: 1
  rope_theta: 3.0
  qk_norm: true
  use_swiglu: true
  use_rope: true
  norm_type: rms
  embed_type: circular_arc
  tie_embed: true
  tie_kv: true
  tie_qo: true
  tie_down_gate: true
  tie_qk_norm: true
  share_norms: true
  share_ln_f: false
  # 🚨 share_ln_f defaults to share_norms (=true), NOT false! Must explicitly set false for 49p.
  # Without share_ln_f: false, you get 46p (ln_f shared with ln1/ln2).
  attn_out_rank: 0
  vocab_size: 10
  lr: 0.01
  min_lr: 0.001
  weight_decay: 0.001
  ohem_ratio: 2.0
  steps: 500000
  batch_size: 128
  warmup_steps: 1000
  curriculum: '3:2000,6:7000,10:rest'
  patience: 500000
  grad_clip: 1.0
  eval_every: 2000
  optimizer: adamw
  commutative_aug: true
  grokfast_alpha: 0.98
  grokfast_lambda: 3.0
  crash_recovery_drop: 0.5
  ```
  **KEY DIFFERENCES from 52p recipes: share_norms: true (saves 3 params), lr: 0.01 (not 0.003), steps: 500000 (49p groks slower), patience: 500000.**
  **PROVEN: idea-900494 (seed 314) reached 97.39% verify at 200K steps. Longer runs (500K+) should push higher.**
  **🚨 DO NOT add `ff_mult`, `carry_bias`, or `tie_gate` — these are NOT in the proven config.**

**🔥🔥🔥 RESEARCH PRIORITIES (UPDATED 2026-04-16T04:45 — PROFESSOR CYCLE) 🔥🔥🔥**

**🔥🔥🔥🔥🔥 49p@100% SOLVED!! SUBMIT FIRST!! 🔥🔥🔥🔥🔥**

**49p@100% is SOLVED and VERIFIED (10010/10010).** Checkpoint: results/cmaes-carry-loss-49p-stable-cuda3/checkpoint.pt.
**submission_49p.py is ready. SUBMIT VIA PR TO anadim/AdderBoard.**

**🚨🚨🚨 STOP ALL 46p EXPERIMENTS — STATISTICALLY DEAD (155 attempts, 0 successes) 🚨🚨🚨**

**🔥🔥🔥 43p UPDATE (2026-04-16T04:45): LEARNING SIGNAL CONFIRMED — λ=2.0 LEADS 🔥🔥🔥**
- **43p (anti_quarter + share_ln_f) is the PRIMARY SUB-49p TARGET.**
- **🔥 idea-bc2974 (seed 1337, λ=2.0): 0.4% accuracy at 18K steps!** Loss=1.01 (well below random). Per-digit mean=51.8%@10K. **STRONGEST early sub-49p signal ever.** λ=2.0 IS BETTER THAN λ=3.0 AT 43p.
- idea-42ce3f (seed 3141, λ=3.0): loss=1.33@22K. Per-digit 53.2%@20K. Learning but no accuracy yet.
- idea-2b8b15 (seed 314, λ=3.0): reset at 16K (divergence). Recovering.
- **CONCLUSION: λ=2.0 should be PRIORITY for 43p. Generate more λ=2.0 seeds.**
- **43p@100% = RANK #4 on trained leaderboard** (between tbukic 41p and 44p).

**🔥🔥🔥 PARAM COUNT TARGETS (UPDATED 2026-04-16T04:45 — PROFESSOR CYCLE) 🔥🔥🔥**
| Config | Params | Rank if 100% | Status |
|--------|--------|-------------|--------|
| anti_quarter + share_ln_f | **43p** | **#4** | **🔥 LEARNING SIGNAL! seed 1337 λ=2.0: 0.4%@18K, loss=1.01. seed 3141 λ=3.0: per-digit 53%@20K.** |
| anti_quarter + share_ln_f + gate_alpha_up + repeat_mix | **39p** | **#2** | **NO SIGNAL yet. idea-561432 (seed 3141): loss=2.3@14K. Very early — let run.** |
| anti_quarter + share_ln_f + repeat_mix + o_tail_scale | **45p** | **#5** (ties tbukic) | 3 in queue. Previous dead (seed 3141@96K). |
| k_alpha_last + sparse_gate | **31p** | **#1** | **DEAD. BLANKET BAN.** |
| k_alpha_last + sparse_gate + repeat_mix + o_tail_scale | **33p** | **#1** | **DEAD. BLANKET BAN.** |
| ~~RotationTransposeTiedLinear (tbukic)~~ | ~~**36p**~~ | ~~**#1 tie**~~ | BLOCKED (code not public). |

**TRACK 0: 39p EXPERIMENTS (HIGHEST PRIORITY — RANK #2 TARGET — INIT BUG FOUND AND FIXED)**
- **39p = anti_quarter + share_ln_f + gate_alpha_up + repeat_mix. VERIFIED 39 params.**
- **🚨🚨🚨 CRITICAL BUG (2026-04-16T01:50): gate_alpha_up with init_gate_alpha=1.0 = tie_gate = DEAD 🚨🚨🚨**
  - ALL 4 running 39p gate_alpha_up experiments are at loss=2.3026 (random) at 48K steps
  - Root cause: gate_alpha=1.0 → silu(1.0*up) = silu(up) = tie_gate (proven DEAD at all param counts)
  - FIX: **init_gate_alpha: 0.1** (or 0.01) — MANDATORY for ALL gate_alpha_up experiments
  - This makes gate start nearly linear, allowing the model to learn the optimal alpha value
  - **ALSO TEST lambda=2.0**: DA found 46p+lambda=2.0 reaches 40.75%. Lambda=3.0 ban was from 49p only.
- **lokimorty's training recipe is UNKNOWN** — gist contains inference code only, no training hyperparameters
- Recipe 39A (CORRECTED — includes init_gate_alpha fix + diverge_patience fix):
```yaml
d_model: 3
n_heads: 1
n_kv_heads: 1
head_dim: 4
ff_dim: 2
n_layers: 1
rope_theta: 3.0
qk_norm: true
qk_norm_type: anti_quarter
use_swiglu: true
use_rope: true
norm_type: rms
embed_type: circular_arc
tie_embed: true
tie_kv: true
tie_qo: true
tie_down_gate: true
tie_qk_norm: true
share_norms: true
share_ln_f: true
gate_alpha_up: true
repeat_mix: true
init_gate_alpha: 0.1
attn_out_rank: 0
vocab_size: 10
lr: 0.01
min_lr: 0.001
weight_decay: 0.001
ohem_ratio: 2.0
steps: 1000000
batch_size: 128
warmup_steps: 1000
curriculum: '3:2000,6:7000,10:rest'
patience: 1000000
diverge_patience: 500000
grad_clip: 1.0
eval_every: 2000
optimizer: adamw
commutative_aug: true
grokfast_alpha: 0.98
grokfast_lambda: 3.0
crash_recovery_drop: 0.5
dead_run_reset: true
```
- Recipe 39B (lambda=2.0 variant — TEST):
  - Same as 39A but `grokfast_lambda: 2.0` and `grokfast_alpha: 0.98`
  - Motivation: DA found 46p+lambda=2.0 reaches 40.75% while lambda=3.0 is dead/slow at sub-52p
- Recipe 39C (smaller init alpha — TEST):
  - Same as 39A but `init_gate_alpha: 0.01`
- **Generate 6 experiments at 39p**: seeds 314, 1337, 17, 4242, 6174, 7777.
  - Mix: 3x Recipe 39A (init_gate_alpha=0.1), 2x Recipe 39B (lambda=2.0), 1x Recipe 39C (init=0.01)
- **WARNING**: lokimorty achieves 99.91% at 39p but their training recipe is unknown. Our recipes are HYPOTHESIS-DRIVEN based on what we've learned at other param counts.

**TRACK 0b: 43p EXPERIMENTS (HIGH PRIORITY — CAN RUN NOW)**
- **43p = anti_quarter_norm + share_ln_f + all existing tying. VERIFIED 43 params.**
- Recipe 43A (EXACT — copy verbatim):
```yaml
d_model: 3
n_heads: 1
n_kv_heads: 1
head_dim: 4
ff_dim: 2
n_layers: 1
rope_theta: 3.0
qk_norm: true
qk_norm_type: anti_quarter
use_swiglu: true
use_rope: true
norm_type: rms
embed_type: circular_arc
tie_embed: true
tie_kv: true
tie_qo: true
tie_down_gate: true
tie_qk_norm: true
share_norms: true
share_ln_f: true
attn_out_rank: 0
vocab_size: 10
lr: 0.01
min_lr: 0.001
weight_decay: 0.001
ohem_ratio: 2.0
steps: 1000000
batch_size: 128
warmup_steps: 1000
curriculum: '3:2000,6:7000,10:rest'
patience: 1000000
diverge_patience: 500000
grad_clip: 1.0
eval_every: 2000
optimizer: adamw
commutative_aug: true
grokfast_alpha: 0.98
grokfast_lambda: 3.0
crash_recovery_drop: 0.5
dead_run_reset: true
```
- **Generate 10 experiments at 43p**: seeds 314, 1337, 17, 13, 4242, 6174, 7777, 2718, 27, 3141.
  - Mix: 5x lambda=3.0, 5x lambda=2.0. **DA found 46p+lambda=2.0 reaches 40.75%. Lambda=3.0 experiments at 43p are very slow (e63961: 2-6% at 448K).**
- Recipe 43B (lambda=2.0 variant):
  - Same as Recipe 43A but `grokfast_lambda: 2.0`
  - Motivation: 46p seed 17 reached 40.75% at 200K with lambda=2.0. Lambda=3.0 may be too aggressive for sub-52p models.
- **WARNING**: share_ln_f=true is statistically dead at 46p. anti_quarter_norm may or may not change the optimization landscape. These experiments are HIGH RISK but HIGH REWARD (rank #4).

**TRACK 1: 45p EXPERIMENTS (HIGH PRIORITY — CAN RUN NOW)**
- **45p = anti_quarter + share_ln_f + repeat_mix + o_tail_scale. VERIFIED 45 params.**
- Recipe 45A = Recipe 43A + `repeat_mix: true` + `o_tail_scale: true` (MUST include `diverge_patience: 500000`)
- repeat_mix gives virtual 2-layer depth which FUNDAMENTALLY changes the forward pass.
- **Generate 5 experiments at 45p**: seeds 314, 1337, 17, 4242, 7777.

**TRACK 2: 31p/33p EXPERIMENTS (🔥🔥🔥🔥🔥 HIGHEST PRIORITY — RANK #1 TARGET 🔥🔥🔥🔥🔥)**
- **🔥🔥🔥🔥🔥 31p CONFIRMED** — `k_alpha_last: true` + `sparse_gate: true` = **31 params (VERIFIED via model instantiation)**. **RANK #1** (below tbukic's 36p!).
  - k_alpha_last: K = alpha * RoPE(QKnorm(Q)), 1p scalar applied AFTER norms/RoPE
  - **CRITICAL: k_alpha_last also makes V=K=Q** — V projection is eliminated (train.py line 312/368). This is WHY 31p not 37p.
  - sparse_gate: gate = silu(a * (x @ b)), rank-1 gate (5p instead of 6p)
  - **RUNNING on GPU 0 (idea-f4c73c, seed 314). More seeds in queue.**
  - **⚠️ RISK**: V=K=alpha*Q is VERY constrained. 31p may be below trainability threshold. But tbukic proves V=Q works at 36p.
- **🔥🔥🔥🔥🔥 33p CONFIRMED** — above + `repeat_mix: true` + `o_tail_scale: true` = **33 params (VERIFIED)**. Also **RANK #1**.
  - repeat_mix adds virtual 2-layer depth (may help overcome V=Q constraint)
  - **RUNNING on GPU 0 (idea-3fcc8f, seed 3141).**
- **⚠️ PREVIOUS CLAIMS OF 37p/39p FOR k_alpha_last PATH WERE WRONG** — V is NOT separate when k_alpha_last=True.
- **DO NOT generate more k_alpha_last experiments until current ones reach 50K steps.** We need signal first.

**TRACK 2b: 39p EXPERIMENTS (gate_alpha_up path — RANK #2 TARGET)**
- **39p CONFIRMED** — gate_alpha_up + repeat_mix + tie_kv = **39 params (VERIFIED)**. Rank #2 tie with lokimorty.
  - This path keeps V=K tying (separate 12p K/V projection), uses gate_alpha (1p) instead of gate_proj (6p).
  - **3 running (seeds 4242, 6174, 7777) + 1 warm-converge (seed with norm_converge_lambda).**
- **gate_alpha_up ban LIFTED for 39p** (ban was based on 52p experiments, invalid for full lokimorty stack).
- **train_39p.py available** for direct training.

**TRACK 3: RE-RUN 43p EXPERIMENTS WITH DIVERGE_PATIENCE FIX (CRITICAL)**
- **idea-e63961 (43p seed 3141): DEAD** — killed by divergence detection bug at 212K. Per-digit was 62.9% (clearly learning). **BUG FIXED: `diverge_patience` config key added. MUST RE-RUN with `diverge_patience: 500000`.**
- **idea-9f5340 (43p seed 17, λ=2.0): DEAD** at 0%.
- **NaN bug killed 7/8 other 43p seeds.** Only 3141 ran with multiplicative fix.
- **REGENERATE 43p experiments with both fixes** (NaN + diverge_patience). Seeds: 3141, 314, 1337, 17, 4242, 6174, 2718, 7777.

**TRACK 4: CODE EVOLUTION FOR 36p (BLOCKED — NOW LESS URGENT)**
- **RotationTransposeTiedLinear** (tbukic, 36p): Code NOT public. BLOCKED.
- If 31p or 33p works, 36p target is MOOT (we'd already beat it).
- These require train.py modifications. Do NOT generate config-only experiments for these.

**CARRY-LOSS RECIPE (for any 90%+ checkpoint at 46p):**
```bash
python3 train_carry_loss.py --checkpoint <ckpt> --seed <seed> --steps 300000 --carry-beta 2.0 --lr 0.001 --device cuda
# Then CMA-ES:
python3 train_cmaes.py --checkpoint <carry-loss-ckpt> --sigma 0.01 --popsize 40 --time-limit 3600 --device cuda
```

**❌ DEAD APPROACHES (UPDATED 2026-04-16T07:00 — PROFESSOR CYCLE):**
- 52p→49p weight projection: DEAD (0.2% best)
- 49p→46p weight projection: EXPECTED DEAD (L2 gap=22.16 between ln1 and ln_f)
- **Recipe H (lambda=2.0) at 49p: DEAD AT ANY LR.** lr=0.01 DEAD. lr=0.003 DEAD (idea-0170e8: 0%@76K, loss=2.3026). **BLANKET BAN for Recipe H at 49p.**
- EGD at 49p: DEAD at any LR. BLANKET BAN.
- All 52p experiments: CLOSED. 49p@100% solved.
- All CMA-ES on 49p: SOLVED. No more needed.
- All NEW 49p experiments: CLOSED. 49p is SOLVED. Let existing sweeps finish only.
- **🚨 ENTIRE k_alpha_last PATH IS DEAD (31p/33p/37p/39p-k_alpha): BLANKET BAN.** V=K=Q too constrained.
  - 31p seed 314: DEAD (0%@186K). idea-f4c73c.
  - 33p seed 3141: DEAD (0%@72K). idea-3fcc8f.
  - **DO NOT generate ANY k_alpha_last or sparse_gate experiments.** All removed from queue.
- **45p seed 3141: DEAD (0%@96K).** idea-24bbc7 killed by dead_run_reset.
- **🚨 SELF-DISTILLATION IS HARMFUL AT SUB-49p: BANNED.** idea-cd1fd5 (43p distill from 49p teacher) had loss=5.27 (2x random). KD loss with T=2.0 and alpha=0.5 overwhelms CE loss at this scale. The T^2 scaling makes KD loss dominate, preventing the student from learning its own algorithm. **Do NOT generate distill_teacher_ckpt experiments.**
- **39p idea-de8b65 (gate_alpha_up, seed 1337, λ=3.0): DEAD at 0%@60K.** dead_run_reset 3/3 exhausted.
- **39p idea-fadeed (gate_alpha_up, seed 314, λ=2.0): DEAD at 0%@32K.** dead_run_reset 3/3 exhausted.
- **43p idea-7d9882 (seed 6174, λ=2.0): DEAD at 0%@34K.** dead_run_reset 3/3 exhausted.
- **45p idea-f9fa4c (repeat_mix, seed 7777): DEAD at 0%@32K.** dead_run_reset 3/3 exhausted.

**DO NOT GENERATE:**
- Any 52p experiments (share_norms: false) — PIVOT COMPLETE.
- Any NEW 49p experiments (share_norms: true, share_ln_f: false) — 49p is SOLVED.
- Any TFT experiments — BANNED.
- Architecture reductions below 46p (d_model<3, ff_dim<1, head_dim<3) — ALL DEAD.
- Removing circular_arc embed — creates 79p, NOT sub-52p.
- **Any k_alpha_last or sparse_gate experiments** — ENTIRE PATH DEAD. BANNED.
- **Any distill_teacher_ckpt experiments at sub-49p** — ACTIVELY HARMFUL. BANNED.

**DEAD PATTERNS:**
- `diverge_patience` no longer required in config — **train.py auto-defaults to 500K when share_ln_f=true** (fixed 2026-04-16T07:00). Including it explicitly is still recommended but no longer a reject criterion.
- `egd: true` at 49p or 46p — DEAD at any LR. BLANKET BAN.
- **carry_loss fine-tune at lr=0.01** — DEAD. Use lr=0.001 or lower.
- **49p→46p weight projection** — EXPECTED DEAD (L2 gap=22.16). Do NOT waste GPU.
- Per-param LR with lambda=3.0 — DEAD
- sphere_norm — DEAD
- perpGrad — DEAD
- **k_alpha_last + sparse_gate (ANY param count, ANY seed)** — DEAD. V=K=Q too constrained. BLANKET BAN.
- **distill_teacher_ckpt at sub-49p** — HARMFUL. Loss 2x random. BANNED.

**❌ SELF-DISTILLATION IS DEAD AT SUB-49p** — idea-cd1fd5 loss=5.27 (2x random). KD loss overwhelms CE at this scale. **BANNED. DO NOT generate distill_teacher_ckpt experiments.**

**PROVEN SEEDS:** 314 (PROVEN at both 52p and 49p). 1337 (PROVEN at 52p, **SHOWING SIGNAL at 43p with λ=2.0**).
**DEAD SEEDS at 49p:** 42, 8. (May be different at 46p — retest allowed.)
**SUB-49p SEED VIABILITY (updated 2026-04-16T04:45):**
- **🔥 43p seed 1337 (λ=2.0)**: 0.4% accuracy@18K. Loss=1.01. Per-digit 51.8%. **#1 PRIORITY EXPERIMENT.**
- **⚠️ 43p seed 3141 (λ=3.0)**: Loss=1.33@22K. Per-digit 53.2%@20K. Learning but noisy.
- **⚠️ 43p seed 314 (λ=3.0)**: Reset at 16K. Recovering.
- **❌ 39p seed 3141 (λ=3.0)**: Loss=2.3@14K (random). Very early.
- **❌ 31p/33p**: BLANKET DEAD.
- **❌ 45p seed 3141**: DEAD at 0%@96K.

## Diversity Budget (MANDATORY — updated 2026-04-16T05:00)
Each batch of 5 ideas MUST include:
- **At least 3 ideas targeting 43p** (anti_quarter + share_ln_f, lambda=3.0) — **CONFIRMED LEARNABLE. PRIMARY TARGET.**
  - Mix seeds: 314, 1337, 4242, 6174, 7777, 2718, 3141
  - Lambda=3.0 is CONFIRMED (e63961 learning). Lambda=2.0 also worth testing (2 ideas in queue).
  - Distillation variant: add `distill_teacher_ckpt` from 49p@100% for 1 in 5 ideas
- At most 1 idea targeting **39p** WITH `init_gate_alpha: 0.1` — rank #2 target, but current experiments dying. HIGH RISK.
- At most 1 idea at **31p/33p or 45p** — deprioritized but not abandoned
- At least 1 idea with steps >= 1000000
- **ZERO 49p experiments** — 49p is SOLVED at 100%.
- **ZERO 46p experiments** — 155/0, same rank as 49p. No competitive value.
- **ZERO 52p experiments** — pivot complete.
- All ideas MUST use `qk_norm_type: anti_quarter` AND `share_ln_f: true`.
- **39p ideas MUST include `gate_alpha_up: true` AND `repeat_mix: true` AND `init_gate_alpha: 0.1`.**
- **NEVER use gate_alpha_up without init_gate_alpha <= 0.1** — default 1.0 = tie_gate = DEAD.
3. **BIPOP-CMA-ES** (new strategy from literature): alternate between large population/sigma for global search and small population/tight sigma for fine local search. More effective than pure IPOP on multimodal landscapes.

**⚠️ YAML CORRUPTION BUG ⚠️**
3/8 newest idea configs have systematic YAML corruption: lines formatted as `'key: value': value`. This produces string keys instead of proper YAML mappings. Code evolution should investigate the idea generation pipeline for this bug.

**⚠️ sphere_norm: FULLY DEAD AND BANNED ⚠️**
66 total experiments, 49 confirmed 0%, 16 still running (all 0%). **Do NOT generate ANY sphere_norm experiments. Add to instant reject list.**

**FOUR valid base recipes. Any idea not matching one of these bases is INVALID:**
- **Recipe E+Lookahead**: alpha=0.99, lambda=3.0, grok_transfer_digits=5, grok_transfer_steps=10000, lookahead_alpha=0.5, lookahead_k=5, batch_size=128, warmup=1000, curriculum "3:2000,6:7000,10:rest", patience=300000
- **Recipe F (E+LA without GrokTransfer)**: alpha=0.99, lambda=3.0, lookahead_alpha=0.5, lookahead_k=5, batch_size=128, warmup=1000, curriculum "3:2000,6:7000,10:rest", patience=300000
- **Recipe H**: alpha=0.98, lambda=2.0, batch_size=128, warmup=1000, curriculum "3:2000,6:7000,10:rest", patience=300000
- **Any of the above + EGD**: Add `egd: true` to any base recipe. **EGD + Recipe H = HIGHEST PRIORITY.**

**VALID RECIPE MODIFIERS (can be added to base recipes):**
- **🚨 MANDATORY FOR RECIPE F: `crash_recovery_drop: 0.5`** — recovers from accuracy crashes. ✅ CONFIRMED CRITICAL. idea-1b949b crashed from 99.60%→10.20% at 56K WITHOUT this — still recovering at 18.40%@60K. idea-1a3611 WITH this recovered instantly. **ALL Recipe F ideas MUST include crash_recovery_drop: 0.5.**
- `grokfast_spike_dampening: true` + `grokfast_spike_threshold: 3.0` + `grokfast_spike_scale: 0.3` — stabilizes training by dampening loss spikes. ✅ IMPLEMENTED. **⚠️ CAUTION: idea-640778 (WITHOUT spike dampening) is FASTER than 1025d7 (WITH it) — 95%@36K vs ~88%@36K. Spike dampening may slow grokking by preventing aggressive exploration. Consider Recipe F + crash_recovery_drop WITHOUT spike dampening as a faster variant.**
- `grokfast_reset_at_acc: 0.05` — resets Grokfast EMA when accuracy first reaches 5%. ✅ IMPLEMENTED. Untested in promising configs.
- ~~`optimizer: muon`~~ — **DEAD at 52p. ALL LRs. DO NOT USE.** (removed from valid modifiers)
- ~~`grokfast_lambda_norm_mult` / `grokfast_lambda_arc_mult`~~ — **DEAD. Per-param Grokfast lambda killed (ideas 3120-3121, 0% at 30K).** (removed)
- ~~`lookahead` with Recipe H (alpha=0.98)~~ — **DEAD. Lookahead incompatible with alpha=0.98.** Lookahead remains valid ONLY with Recipe F and Recipe E+Lookahead (alpha=0.99).

**🔥🔥🔥 CMA-ES POST-TRAINING REFINEMENT — MANDATORY FOR ANY 49p MODEL AT 99%+ 🔥🔥🔥**
CMA-ES PROVEN at 52p: 99.96% → 100.0% in 6 seconds (sigma=0.001). Apply to ANY 49p checkpoint reaching 99%+.
```bash
python3 train_cmaes.py --checkpoint results/<idea-id>/checkpoint.pt --sigma 0.001 --popsize 40 --device cuda
```
If CMA-ES alone doesn't reach 100%, try interpolation + CMA-ES:
```bash
python3 train_interpolate.py --checkpoints ckpt1.pt ckpt2.pt --cmaes-refine --sigma 0.001 --device cuda
```

**TARGETED FT STILL INCLUDED as gradient-based fallback (TFT may help gradient training reach higher base accuracy before CMA-ES):**
```yaml
targeted_ft_rounds: 20       # 20-30 rounds
targeted_ft_lr: 0.0003       # low LR for fine-tuning (0.0001 for Recipe B)
targeted_ft_steps: 500        # steps per round
targeted_ft_search_size: 50000  # samples to search for wrong examples
targeted_ft_lr_decay: 0.8     # decay LR 20% each round
targeted_ft_wrong_frac: 0.6   # 60% of batch from wrong examples
```

## Your Task
1. Read `{results_dir}/report.md` for current experiment results
2. Read `{results_dir}/status.json` for pipeline status
3. Analyze what worked and what didn't
4. Generate new experiment ideas and append them to `{ideas_file}`

## Current State
- Research cycle: {cycle}
- Completed experiments: {completed}
- Queued experiments: {queued}

## Competition Context (UPDATED 2026-04-15 — PROFESSOR CYCLE)
- Current trained #1: **36 params** (tbukic, K=rot(Q) 1p, V=Q, down=rot(up^T) 1p, all norms shared, tied QK norms)
- Current #2: **39 params** (lokimorty, 99.91% — Anti-Quarter QK norm, repeat-mix shared block)
- **52p at 100% exists on leaderboard** (vijec, rank #6 — uses **Grokfast paper defaults: alpha=0.98, lambda=2.0** + iterated targeted FT)
- **🔥🔥🔥 PIVOT TO 49p (2026-04-15).** ALL new experiments target 49p (share_norms: true). 52p is CLOSED.
- **49p best: 49 params at 97.39% verify** (idea-900494, share_norms=true, lr=0.01, lambda=3.0, alpha=0.98, seed 314). Only 261 wrong/10010. **1.61% from qualifying.**
- **52p best (REFERENCE ONLY): 52 params at 99.96% verify** (idea-9a5ca1). Hard ceiling. Not pursuing further.
- **4 seed sweeps running. idea-c1c6f4 queued. ~3991 completed. 4 CMA-ES runs active (finishing soon).**
- **External leaderboard STABLE** (re-verified 2026-04-16 via WebFetch). 29 entries. No changes since March 26.
- **49p@99%+ would rank ABOVE vijec's 52p@100% on the leaderboard (fewer params wins).**
- **EGD at 52p = DOES NOT HELP (99.94% vs 99.96%). At 49p with lr=0.01 = DEAD. May work at 49p with lr=0.003 — UNTESTED.**
- **egd_off_at_acc IMPLEMENTED in train.py** — UNTESTED at 49p. Try with 49p recipes.
- **DEAD SEEDS at 52p: 7, 13, 27, 37, 42, 73, 99, 101, 123, 271, 512, 2025, 8888.** These are NOT necessarily dead at 49p — 49p is a different optimization landscape. Resweep at 49p.

### 🔥🔥🔥🔥🔥 49p IS NOW THE PRIMARY TARGET 🔥🔥🔥🔥🔥

**idea-900494 PROVED 49p is viable:** 49 params, 97.39% verify accuracy (9749/10010 correct).
- Config: `share_norms: true, grokfast_lambda: 3.0, grokfast_alpha: 0.98, lr: 0.01, min_lr: 0.001, seed: 314, steps: 200000, weight_decay: 0.001`
- Training was volatile (accuracy swings between 0% and 98% between eval points) — classic pre-grokking instability
- **Only 261 wrong out of 10010!** 1.61% from qualifying.
- **NEEDS: Longer training (500K-1M), crash_recovery_drop=0.5, seed sweeps.**

**ALL new experiments MUST use share_norms: true. 52p experiments are BANNED.**

### DEAD DIRECTIONS (UPDATED 2026-04-16T04:15 — PROFESSOR CYCLE)

- **ALL 52p experiments (share_norms: false)** — BANNED. Pivot complete.
- **ALL 49p experiments (share_norms: true, share_ln_f: false)** — BANNED. 49p is SOLVED.
- **ALL 46p experiments (share_ln_f: true WITHOUT anti_quarter)** — BANNED. 155 experiments, 0 successes. STATISTICALLY DEAD.
- **ALL TFT variants** — EXHAUSTED. 7+ complete. BANNED.
- **EGD at any sub-52p** — DEAD. BLANKET BAN.
- **sphere_norm** — DEAD AND BANNED (66 experiments).
- **perpGrad** — DEAD AND BANNED.
- **Muon optimizer** — DEAD AND BANNED.
- **Architecture reductions (d_model<3, ff_dim<1, head_dim<3)** — ALL DEAD.
- **Removing circular_arc** — Creates 79p, NOT sub-52p. BANNED.
- **grad_accum_steps > 1** — Large batch kills grokking. DEAD.
- **grokfast_type: ma or dual** — ALL DEAD. BANNED.
- **grokfast_per_param_lambda** — DEAD. BANNED.
- **per-param LR + lambda=3.0** — DEAD (28/28 at 0%).
- **k_rot_q WITHOUT sphere_norm** — DEAD (100+ at 0%).
- **lookahead with alpha=0.98** — DEAD.
- **Recipe H (lambda=2.0) at sub-52p** — BLANKET DEAD at any LR.

### GPU STATUS (UPDATED 2026-04-16T04:15 — PROFESSOR CYCLE)
**4 experiments running (3 stale 49p + 1 46p). All GPUs at 24-25% utilization.**
- GPU 0: idea-121da0 (49p Lookahead seed 314 — stale but novel technique test)
- GPU 1: idea-0157e8 (49p/46p seed 17 1M — only seed to show life at 46p)
- GPU 2: CMA-ES (carry-loss, 10009 stuck) + seed sweep  
- GPU 3: CMA-ES (carry-loss, 10009 stuck) + seed sweep
**CMA-ES runs will finish within ~35 min. GPUs will then be available for more seed sweeps.**
**Generate seeds 4242, 6174, 7777, 2718 NOW so they're queued when GPUs free up.**

### LEADERBOARD INTELLIGENCE (verified 2026-04-11T16:00 — PROFESSOR CYCLE)
1. **vijec's 52p = 100% (rank #6)**: Uses alpha=0.98, lambda=2.0 + iterated targeted FT. **idea-057702 confirms: 300K steps = DEAD (80.5%). vijec likely trains 500K-1M+.**
2. **lokimorty's 39p = 99.91% (rank #2)**: AntiQuarterNorm + RepeatMixBlock. **FULL CODE OBTAINED.** Code evolution candidate.
3. **tbukic's 36-44p entries**: Need RotationTiedLinear code evolution.
4. **evindor's 57p = 100% (rank #8)**: Parametric circ embed, tied V/O, tied Q/K+phase, rank-1 out. Different approach.
5. **No new entries since March 26.** Leaderboard frozen. We have time but MUST FILL THE EMPTY QUEUE.
6. **Muon optimizer: ALL DEAD at 52p.** BANNED. AdaMuon unlikely to help at this scale.
7. **Commutator defect (arxiv 2602.16967)**: Early-warning signal. CODE EVOLUTION CANDIDATE for early-stopping dead runs.
8. **NEW: arxiv 2604.04655** "Grokking as Dimensional Phase Transition" (April 2026). Seed-dependent grokking timing varies >2x. VALIDATES seed sweep strategy.

### FEATURES IN train.py
**⚠️ CRITICAL: Always use `ff_dim: 2` in configs, NOT `ff_mult: 2` alone.**
**Working:**
- `ff_dim: 2` — **MANDATORY.** Sets FFN hidden dim to 2 for 52p. Without this, `ff_mult: 2` creates ff_dim=6 (76p model).
- `lr_norm_mult`, `lr_arc_mult`, `lr_up_mult` — per-param LR. Full set (norm+arc+up) ONLY works with lambda=4.0 (Recipe B). **NEW: `lr_arc_mult` ALONE with Recipe A is UNTESTED.** arxiv 2505.15624v1 shows 10x higher embedding LR accelerates grokking. Try lr_arc_mult=3.0 with Recipe A (lambda=3.0).
- `ohem_ratio: 2.0` — REQUIRED for all experiments.
- `sphere_norm: true` + `sphere_tau: 10.0` — Spherical residual stream. **CAUSES 8K COLLAPSE in most seeds. Use selectively.**
- `commutative_aug: true` — Commutative data aug. Zero params. Include in all new ideas.
- ~~**`grok_transfer_digits` + `grok_transfer_steps`**~~ — GrokTransfer. **DEAD with Recipe H.** 99.82% verify at 76p (idea-2144) but COUNTERPRODUCTIVE at 52p with alpha=0.98. DO NOT USE with Recipe H.
- **`lookahead_alpha` + `lookahead_k`** — Lookahead optimizer. **NEVER TESTED at true 52p with Recipe A.** Only tested at 76p (ff_mult bug) or with old recipe (stuck at 50%). Use alpha=0.5, k=5.
- `ckpt_avg_k` — Checkpoint averaging. idea-2132 grokking at 27.2% at 96K with ckpt_avg_k=10.
- `sam_rho` — SAM optimizer. rho=0.1 DIVERGED. LOW PRIORITY.

**DEAD/HARMFUL — DO NOT USE:**
- `perp_grad: true` — HARMFUL. ALL experiments 0%. BANNED.
- `lr_norm_mult` + `grokfast_lambda: 3.0` — DEAD combo. 28/28 at 0%.
- `sphere_norm` + Recipe B (per-param LR + lambda=4.0) — DEAD (idea-2131).

**STILL NEEDING code evolution:**
- **RotationTransposeTiedLinear** — down=rot(up^T). Enables 36p. Code NOT public.

**ALREADY IMPLEMENTED (corrected 2026-04-16T01:00):**
- RepeatMixBlock: IMPLEMENTED ✓ (`repeat_mix: true`)
- k_alpha_last: IMPLEMENTED ✓ (K = alpha * RoPE(QKnorm(Q))). Key to 37p.
- sparse_gate: IMPLEMENTED ✓ (rank-1 gate). Key to 37p.
- gate_alpha_up: IMPLEMENTED ✓ (gate = alpha * up). Key to 39p path A.
- o_tail_scale: IMPLEMENTED ✓. Part of 45p and 39p path B.

## What We Know Works
1. **Architecture**: d=3, hd=4, ff_dim=2, SwiGLU, circular arc embed, RoPE theta=3
2. **Tying that trains**: tie_kv, tie_qo, tie_qk_norm, tie_down_gate
3. **Tying that KILLS training**: tie_gate (gate=up), tie_qk (K=Q=V)
4. **share_norms: true = 49p** — PROVEN at 97.39% verify (idea-900494). NOW THE PRIMARY TARGET.
5. **RECIPE A — RELIABLE GROKKING (NO per-param LR)**:
   ```yaml
   grokfast_alpha: 0.99, grokfast_lambda: 3.0, lr: 0.003, min_lr: 0.0003, weight_decay: 0.001, ohem_ratio: 2.0, steps: 300000
   ```
   → d40378/dc38e2 at 96.6% at 160K. Needs 300K to stabilize at 99%+.
6. **RECIPE B — VOLATILE GROKKING (per-param LR + λ=4.0)**:
   ```yaml
   grokfast_alpha: 0.99, grokfast_lambda: 4.0, lr: 0.003, min_lr: 0.0003, weight_decay: 0.001, ohem_ratio: 2.0, lr_norm_mult: 3.0, lr_arc_mult: 0.5, lr_up_mult: 1.5, steps: 200000
   ```
   → idea-2104 peaks 99.8% but crashes every 10-20K steps. Consider 300K for stability.
7. **Encoding**: [0] + rev(a,10) + [0,0] + rev(b,10) + [0] = 24 tokens
8. **Multiphase pipeline**: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
9. **Seed 42 is the grokking seed.** Seed 1337 also groks with Recipe H. Seed 314 is our verified best for Recipe B (99.62%) but DEAD for Recipe H.
10. **OHEM ratio 2.0 is REQUIRED** for 52p→100% push.
11. **Lambda=3.0 is the ONLY working lambda with OHEM (without per-param LR)** — for Recipe A.
12. **Lambda=4.0 + per-param LR + OHEM** is the ONLY working per-param LR combo — Recipe B.
13. **🔥 RECIPE H = THE BEST RECIPE** — alpha=0.98, lambda=2.0, targeted_ft. Groks 2-3x faster than A or B. Two seeds confirmed (42 at 11% at 36K, 1337 at 3.2% at 38K).
14. **perpGrad is HARMFUL** — BANNED.
15. **Per-param LR + lambda=3.0 = DEAD** — 28 experiments, ALL 0%.
16. **Lambda=2.0 + alpha=0.99 = DEAD** — 11+ experiments. **Lambda=2.0 + alpha=0.98 = Recipe H = WORKING.**
17. **New paper (arxiv 2602.16967)**: Amplifying gradient non-commutativity accelerates grokking 32-50%. OHEM + commutative_aug already do this. Theoretical support for our approach.

## EXHAUSTED APPROACHES — DO NOT PROPOSE THESE
1. **embed_type=none / embed_type=zero / embed_type=identity** — BUG: produces 79-115p models, NOT the expected 49p.
2. **Scalar norms (norm_scalar=true, ln1_type=scalar)** — 12+ experiments, ALL 0%.
3. **d=2, hd=2** — untrainable at any param count.
4. **k_rot_q at 41p** — 100+ experiments, ALL 0%. DEAD without spherical residual stream. DO NOT GENERATE.
5. ~~share_norms=true WITHOUT adaptive_wd~~ — Was dead at 52p but NOW PROVEN at 49p (idea-900494, 97.39%). share_norms=true IS the 49p target. The old 155 experiments used wrong hyperparameters.
6. **tie_qk=true + share_norms=true** — 43 experiments, ALL 0%. DEAD.
7. **tie_qk=true + tie_qk_norm=true** — All experiments 0%. DEAD.
8. **tie_qk=true at 50p (ANY config)** — Clean test completed: loss stuck at 2.3, 0% acc. K=Q=V can't learn. DEAD.
9. **tie_gate (gate=up in SwiGLU)** — always 0%.
10. **No curriculum** — always 0% for sub-100p models.
11. **Seed 8 sweeps on 58p configs** — done 20+ times, all ~97.8%.
12. **ff_dim=3** — accuracy drops to 88-95%.
13. **Removing ln_f** — 71.6% best. Too destructive alone.
14. **Cayley K / Householder K / k_rot_givens** — all 0%.
15. **k_bias_q / k_diag_q** — underperform (<93%).
16. **Pure lambda=2.0 single-phase** — 97.60% max.
17. **52p without OHEM at 200K steps** — 3000+ experiments done. Max 99.26% verify. OHEM required.
18. **head_dim=3 or head_dim=2** — not competitive.
19. **d_model=2 or d_model=4** — d=3 is the proven frontier.
20. **OHEM 300K-500K step runs (seed 314, lambda=3.0, standard recipe)** — 77 experiments, NONE grokked to 99%+. Best 90.75%. EXHAUSTED.
21. **k_alpha_q (K=alpha*Q)** — ideas 2001-2010, ALL 0%. DEAD.
22. **v_eq_q (V=Q)** — ideas 2001-2010, ALL 0%. DEAD.
23. **gate_alpha_up (gate=alpha*up)** — ideas 2001-2010, ALL 0%. DEAD. (idea-2004 was clean test, 0%)
24. **Any combination of k_alpha_q / v_eq_q / gate_alpha_up** — 10 experiments at 33-49p, ALL 0%. DEAD.
25. **perp_grad: true** — ALL experiments using perpGrad at 0% (ideas 2101-2103, 2107-2110). perpGrad suppresses grokking. HARMFUL.
26. **adaptive_wd: true** — ideas 2105/2106/2109, ALL 0%. adaptive_wd itself is DEAD. (share_norms works fine WITHOUT adaptive_wd — see idea-900494 at 97.39%)
27. **grokfast_lambda=4.0 + ohem_ratio=2.0** — 4/4 experiments (ad2b44, 5c75b6, 2332d1, a90c27) ALL stuck at loss=2.3026 (random) at 40-54K steps. Lambda=4.0 only works WITHOUT OHEM. With OHEM, MUST use lambda=3.0.
28. **Lambda=2.0 + standard recipe (lr=0.01, wd=0.01)** — idea-064359 COMPLETED at 0.4% accuracy after 300K steps. DEAD.
29. **grokfast_lambda=4.0 + per-param LR + OHEM = VOLATILE BUT WORKS** — idea-2104 peaks 99.8% at 154K/200K. This is Recipe B. Works ONLY at lambda=4.0.
30. **Per-param LR + lambda=3.0 + OHEM** — 28 experiments (seeds 42, 314, 27), ALL at 0% after up to 98K steps. Per-param LR DISRUPTS grokking at lambda=3.0. DEAD.
31. **Lambda=2.0 + alpha=0.99 + alt-recipe LR (lr=0.003, wd=0.001)** — 11 experiments, ALL dead after 50K+ steps. **BUT: vijec uses alpha=0.98 (not 0.99) + lambda=2.0 + targeted FT. The alpha=0.98 combination is UNTESTED. See Recipe H.**
32. **k_rot_q at 41p (ANY recipe, WITHOUT sphere_norm)** — 100+ experiments, ZERO accuracy signal. Standard recipe: loss=2.3026. Alt recipe: loss 1.9-2.0 but 0%. Per-param LR: 0%. DEAD without sphere_norm.
33. **sphere_norm WITHOUT sphere_norm_unembed (pre-fix)** — idea-2121/2122/2124 all COLLAPSED (Softmax Collapse). Loss drops initially then jumps to random (2.3026). Unembedding weight normalization is REQUIRED per arxiv 2603.05228. FIX DEPLOYED: sphere_norm_unembed now auto-enabled.
30. **Alt recipe (lambda=3.0, OHEM 2.0) at 200K steps, seed 42** — idea-0b3f75 COMPLETED at 97.78% verify (98.10% acc). Recipe gets to 98% batch acc but 200K insufficient to stabilize at 99%+. Use 300K steps.

## Parameter Budget

**52p baseline (REFERENCE):**
- embed: 3p (circ arc A, start, stride)
- Q proj: 12p (4×3, also O=Q^T)
- K proj: 12p (4×3, also V=K)
- QK norm: 4p (shared between q_norm and k_norm)
- gate_proj: 6p (2×3)
- up_proj: 6p (2×3)
- down_proj: 0p (tied to gate^T)
- ln1: 3p, ln2: 3p, ln_f: 3p
- lm_head: 0p (tied to embed)
- **Total: 52p**

**49p (share_norms=true) — CURRENT TARGET:**
- Same as 52p but ln1=ln2=ln_f shared → saves 6p, adds back 3p shared = net -3p
- **Total: 49p**


## THINKER PROPOSALS (2026-04-15 20:56 UTC)

### KEY FINDING: 52p→49p Weight Projection Is DEAD
Direct weight projection from 100% 52p → 49p gives **0.01% accuracy**. The 52p model's three
norms are architecturally incompatible with sharing. The 49p model uses a fundamentally DIFFERENT
algorithm. Do NOT waste GPU on warm_share_norms_step from 52p checkpoints.

### NEW: Carry-Chain-Weighted Loss (train_carry_loss.py)
Weights per-digit CE loss by carry chain length. Positions at the end of long carry chains
(the ones the 49p model gets wrong) get 1.5-3x more gradient signal. Compatible with OHEM.
Also includes carry-biased OHEM: select samples by structural difficulty, not just loss.

**Launch commands:**
```bash
# Carry-aware loss, seed 314 (proven 49p seed):
python3 train_carry_loss.py --seed 314 --steps 500000 --carry-beta 2.0 --device cuda

# Carry-aware loss + carry-biased OHEM:
python3 train_carry_loss.py --seed 314 --steps 500000 --carry-beta 2.0 --carry-ohem --device cuda

# Seed 1337 (proven 52p seed, untested at 49p):
python3 train_carry_loss.py --seed 1337 --steps 500000 --carry-beta 2.0 --device cuda

# Fine-tune existing 49p checkpoint with carry loss:
python3 train_carry_loss.py --checkpoint results/idea-900494/checkpoint.pt --seed 314 --steps 300000

# With Lookahead optimizer:
python3 train_carry_loss.py --seed 314 --steps 500000 --carry-beta 2.0 --lookahead --device cuda
```

### DEAD: train_project_49p.py (52p→49p projection)
Implemented and tested. Projection + CMA-ES on norm params. Result: 0%. DEAD.

## THINKER PROPOSALS (2026-04-16 00:44 UTC) — PARADIGM SHIFT

### CRITICAL FINDING: 39p IS ALREADY BUILDABLE — NO CODE EVOLUTION NEEDED
**The "sparse_gate" and "k_alpha_last" labels in GOAL.md are WRONG.**
39p = 43p (anti_quarter + share_ln_f) + gate_alpha_up (-5p) + repeat_mix (+1p) = **39 params**.
ALL features already exist in train.py. Verified: `AdderTransformer(cfg).count_params() == 39`.

**gate_alpha_up was BANNED at 52p** (ideas 2001-2010, ALL 0%) but those experiments used it
in ISOLATION at 52p without anti_quarter, repeat_mix, or shared norms. At 39p with the full
lokimorty architecture, it's a completely different optimization landscape.
**lokimorty achieves 99.91% at 39p — rank #2 on the leaderboard.**

**The gate_alpha_up ban DOES NOT APPLY to 39p with the full lokimorty stack.**

### NEW: train_39p.py (39p lokimorty-architecture training)
Direct training script for 39p using all existing infrastructure.

**Launch commands:**
```bash
# Direct 39p training (shared norms from step 0):
python3 train_39p.py --seed 314 --steps 1000000 --device cuda

# With norm convergence regularization (start separate, converge gradually):
python3 train_39p.py --seed 1337 --steps 1000000 --warm-converge --device cuda

# Seed sweep:
for s in 314 1337 17 4242 6174 7777; do
  python3 train_39p.py --seed $s --steps 1000000 --device cuda &
done

# If 39p reaches 90%+, apply carry-loss → CMA-ES:
python3 train_carry_loss.py --checkpoint results/39p-<seed>/checkpoint.pt --steps 300000
python3 train_cmaes.py --checkpoint results/<carry-loss>/checkpoint.pt --sigma 0.01 --popsize 40
```

### NEW: Norm Convergence Regularization (train.py)
Added `norm_converge_lambda` and `norm_converge_ramp` config keys.
Adds L2 penalty between ln1, ln2, ln_f during separate-norm training phase.
Forces norms to converge gradually before hard merge, avoiding the destructive
"L2 gap=22.16" hard switch.

**Config keys:**
```yaml
norm_converge_lambda: 0.5    # L2 penalty strength (0=disabled; 0.1-1.0 typical)
norm_converge_ramp: 80000    # linearly ramp penalty from 0 to lambda over N steps
warm_share_norms_step: 100000  # hard merge after convergence
```

### 39p PARAM BREAKDOWN (VERIFIED)
| Component | Params | Notes |
|-----------|--------|-------|
| arc_A, arc_start, arc_stride | 3 | Circular arc embedding |
| q_proj (4x3) | 12 | Also O=Q^T |
| k_proj (4x3) | 12 | Also V=K |
| q_norm (anti-quarter) | 1 | Shared with k_norm |
| gate_alpha | 1 | gate = alpha * up (replaces 6p gate_proj) |
| up_proj (2x3) | 6 | Also down=gate^T (but gate is scalar) |
| ln1 (shared) | 3 | = ln2 = ln_f |
| repeat_gain | 1 | Virtual 2-layer interpolation |
| **TOTAL** | **39** | **Rank #2 if 99%+** |

### PRIORITY UPDATE
1. **39p experiments should run IN PARALLEL with 43p/45p**, not replace them
2. **39p@99%+ would be rank #2** (only behind tbukic's 36p)
3. **The 3-step pipeline applies**: gradient training → carry-loss → CMA-ES
4. **gate_alpha_up ban is LIFTED for 39p with full lokimorty stack only**
