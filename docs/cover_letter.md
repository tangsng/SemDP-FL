# Cover Letter

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled "**SemDP-FL: Semantic-Sensitivity-Driven Adaptive Differential Privacy for Federated LoRA Fine-Tuning of Large Language Models**" for consideration as an original research article in the *Journal of Information Security and Applications* (JISA).

**Motivation and problem.** Federated fine-tuning with Low-Rank Adaptation (LoRA) allows distributed clients to adapt a shared large language model without exposing raw text, yet the exchanged low-rank updates remain vulnerable to gradient-inversion attacks. Client-level differential privacy (DP) is the standard countermeasure, but existing DP federated LoRA methods inject noise *uniformly* across all trainable parameters — overlooking the pronounced heterogeneity of semantic sensitivity across transformer layers, and thereby under-protecting the most leakage-prone parameters while over-perturbing the rest.

**Our contribution.** We propose SemDP-FL, which allocates a fixed client-level DP budget across LoRA parameter groups according to a fused semantic sensitivity score (an LLM-elicited prior combined with gradient evidence measured on a small public proxy set). The most sensitive groups receive enforced noise floors, and the residual budget is distributed by a closed-form, utility-optimal water-filling rule under exact sampled-Gaussian Rényi-DP accounting. We further (i) prove an explicit sub-optimality bound for uniform noise allocation (a measured 2.08× distortion ratio computable *a priori*), and (ii) establish a Gaussian-channel mutual-information bound that turns the tier floors into hard leakage caps, corroborated by a strengthened head-gradient-inversion probe (n = 100).

**Key results.** On federated SST-2 and AG News fine-tuning of Qwen2.5-0.5B with 100 non-IID clients (five seeds; four recent baselines — DP-FedAvg, DP-AdaptClip, DP-FedSAM, FFA-LoRA+DP — reimplemented on a common codebase with identical accounting), SemDP-FL improves mean accuracy by 5.8 percentage points over the strongest baseline at ε = 4 (7.7 over DP-FedAvg), by 25.5 points on AG News with statistical significance (p = 0.012), and recovers up to 86% of the non-private accuracy. The evidence base comprises 182 training runs with exact per-run ε accounting; negative and tied outcomes (e.g., a tie with FFA-LoRA+DP at ε = 8) are reported in full.

**Fit to JISA.** The manuscript falls squarely within the journal's scope of "differential privacy techniques" and "federated learning privacy assurance," combining a formal privacy analysis (Rényi-DP accounting, mutual-information bounds) with an extensive, statistically disciplined experimental study — the balance of theory and thorough comparative experiments that JISA readers expect.

This manuscript is our original work, has not been published previously, and is not under consideration by any other journal. All authors have approved the submission and declare no competing financial interests or personal relationships. The work is supported by the Hebei Provincial Science and Technology Program Project (Grant No. 25360301D).

Thank you for your consideration. We look forward to the reviewers' feedback.

Sincerely,

Song Tang^1,2,3^, Zhigang Jin^1,\*^, Zhiqiang Wang^2,3^

^1^ School of Electrical and Information Engineering, Tianjin University, Tianjin 300072, China
^2^ Institute of Applied Mathematics, Hebei Academy of Sciences, Shijiazhuang 050081, China
^3^ Information Security Authentication Technology Innovation Center of Hebei Province, Shijiazhuang 050081, China

^\*^ Corresponding author. E-mail: zgjin@tju.edu.cn
