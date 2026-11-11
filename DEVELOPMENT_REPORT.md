# Development report / 开发报告

## 2026-11-11: Requirements and preflight
- Read all three pages of the original CIE6007 2026–27 T1 guidelines. Main report A4, two columns, >=10 pt, <=4 pages excluding figures/tables; supplementary commented code; name/student ID; references and AI acknowledgement. Deadline is only in Blackboard, not verified here.
- All writes confined to /volume/yzhao04. Compute, tests, plots and PDF build execute on the paired Shanghai GPU node.
- Fresh SSH and shared GPFS checks succeeded. Four H200 GPUs; driver 580.105.08; PyTorch 2.11.0+cu130; torchvision 0.26.0+cu130; Triton 3.6.0; Helion absent. Native torch.optim.Muon exists and its signature was inspected.
- Existing keep-alive PID 3979182 occupies all four GPUs. Requested permission to move it to GPUs 1–3 and restore afterward; pending. Do not silently stop it.
- Context7 queried against PyTorch v2.11.0, supplemented with installed source and version-specific official documentation.
- Original Muon README retrieved through GitHub connector; official blog and Adam/AdamW/Fashion-MNIST papers verified. arXiv:2610.21555 exists, v1, 15 October 2026, Akhtar et al.; its findings concern different architectures and are not our measurements.
- Author identity requested, pending. Public artifacts will be anonymous. GitHub public write remains pending review and explicit confirmation.

## Protocol fixed before data inspection
- Fashion-MNIST official train split: stratified 50,000 training / 10,000 validation, split seed 2026; official 10,000 test separate and unopened by tuning/training.
- MLP 784 -> 256 -> 256 -> 256 -> 10, ReLU; float32, no dropout/augmentation, pixel scaling [0,1]. Batch 128; no schedule, 15 epochs final, 10 epochs tuning.
- SGD momentum .9; Adam and AdamW betas (.9,.999), eps 1e-8. All weight matrices decay 1e-4; biases decay zero. SGD/Adam use coupled L2; AdamW/Muon use decoupled decay. Equal numeric decay does not imply equal effective regularization.
- Muon only two hidden-to-hidden 256x256 matrices; input layer, output layer and all biases use AdamW. Momentum .95, Nesterov, five Newton–Schulz iterations, original LR scaling. Auxiliary AdamW LR .001 fixed in advance, not tuned separately.
- Three candidates each: SGD [.01,.03,.1]; Adam/AdamW [.0003,.001,.003]; Muon [.005,.02,.05]. Tuning seed 17; select highest validation accuracy over 10 epochs (tie: lower validation loss, then lower LR). Final seeds 101,202,303; paired initial weights and minibatch order.
- Report final-epoch checkpoints (fixed steps) and validation-best checkpoints (primary test comparison). Test only after all training/configuration frozen.
- Predefined threshold validation accuracy .85. Fixed-time readout uses last complete epoch available at 20 seconds cumulative synchronized train+validation time, with validation-best checkpoint so far. Report unavailable if a trajectory does not span this budget.
- Warm up 20 steps on disposable model and fresh optimizer, then reseed/recreate both. CUDA synchronize before/after train and evaluation. Dataset resident on GPU; exclude loading, warmup, checkpoint I/O from train time; report train+validation time separately. No compilation/autotuning benchmarks or energy claims.
- Run sequentially on GPU 0; randomized optimizer order within seed to reduce order bias. Record individual epoch rows, final evaluation rows, source/config hashes and hardware/software.

## Implementation and resource isolation
- User authorized keep-alive adjustment. Terminated only the identified PID 3979182 and launched the unchanged keep-alive with KEEP_ALIVE_GPUS=1,2,3 (PID recorded outside public repo). Fresh nvidia-smi confirms GPU 0 empty and GPUs 1–3 protected. Restore all four after experiments.
- Received author identity; stored in ignored, mode-600 paper/identity.tex, never in tracked source.
- Created a project virtual environment on the volume. Reused the existing CUDA runtime packages through a documented .pth file; installed pytest/PyYAML into the new environment from downloaded wheels. Clean portable install instructions will use pinned requirements.
- Downloaded the four official Fashion-MNIST gzip files from the author's repository; loaders verify official MD5 checksums. Test file is stored but its tensors are not loaded by training/tuning.
- Implementation separates tuning, final training, and final test evaluation. Per-run checkpoints and CSVs enable resume at run boundaries. Unmodified native PyTorch optimizers, explicitly disabled foreach for SGD/Adam/AdamW; no compilation.

## Smoke validation and pre-training budget amendment
- 12 pytest cases passed (CPU and CUDA updates/repeatability for all four optimizers, routing, splits, CSV, coupled-vs-decoupled decay). All four two-epoch smoke jobs completed on 512 training samples; smoke metrics explicitly tagged and excluded from research tables.
- Smoke timing indicates 20 seconds could exceed a full 15-epoch baseline run. Before any full-data tuning/final run, changed the predeclared common train+validation budget to **5 seconds**. This is a retrospective epoch-boundary budget readout of fixed-epoch trajectories, not a separately stopped wall-clock run. Actual samples/epochs at budget will be reported.
- Tectonic 0.15.0 runs on GPU node; GPU outbound networking unavailable. Download TeX bundle on CPU/shared volume and compile on GPU.

## Completed experiment and audit
- Frozen experiment code commit: `3a281c2`; all per-run metadata record the full commit plus source/configuration SHA-256.
- Completed 12 balanced tuning runs (3 rates × 4 methods × 10 epochs), then 12 formal runs (3 seeds × 4 methods × 15 epochs); 308 epoch rows including 8 smoke rows.
- Selected LRs by validation only: SGD .03, Adam .001, AdamW .001, Muon .005. Muon's winner is at the lower search boundary; this is a limitation, not a reason to retune after viewing test results.
- Only after all formal training, evaluated the 36 predeclared best/final/time-budget checkpoints on the official test split.
- Validation-selected test accuracy, mean ± sample SD (%): SGD 87.57 ± .32; Adam 88.32 ± .40; AdamW 88.59 ± .18; Muon hybrid 89.04 ± .29.
- Mean total train seconds: 9.40, 11.31, 11.16, 22.04 respectively. Muon/AdamW time ratio 1.97; paired accuracy difference +.453 ± .351 percentage points (n=3; no significance claim).
- Five-second retrospective train+validation readout mean test (%): 86.94, 87.21, 87.54, 87.18. This is explicitly not a strict end-to-end timed stop.
- `validate_results.py` passed: balanced budget, paired initial hashes, run lengths, cumulative timing sums, all checkpoint hashes, validation-best/time-budget selection, raw-to-summary mean and sample-SD recomputation.
- Restored keep-alive on GPUs 0,1,2,3 after final evaluation. Fresh log confirms all four active. No other existing workload was terminated.
- First LaTeX build succeeded: 4 main pages including figures/tables/references + 4 supplementary pages. Visual review and final polishing in progress. Tectonic 0.15.0 and downloaded v33 resource bundle reside on the mounted volume; compilation performed on GPU node.
- GitHub CLI and connector identify account XFDG. Proposed XFDG/cie6007-optimizer-benchmark returned 404 (no accessible existing repository); no remote write performed. Public release remains pending explicit confirmation after artifact review.

## Reproduction review fix (after measurement)
- Found a fresh-clone resume issue during review: published completed-run metadata could cause training to skip even though checkpoints are intentionally untracked. Fixed resume to require every expected checkpoint and matching configuration/epoch count; otherwise regenerate the run. Final evaluation now fails clearly if a required checkpoint is absent.
- Added a regression test that deletes a completed run's checkpoint, verifies regeneration, and rejects incompatible cached configuration. This orchestration-only change does not alter the measured update/epoch loop. Historical measurements and their original source hashes remain unchanged; measured source remains retrievable at `3a281c2`.
- Supplementary snippets now derive directly from source markers, preventing stale line-number excerpts after maintenance.

## Final local acceptance
- Final PDF: **4 main pages + 3 supplementary pages**, all A4; body and code use 10 TeX pt (9.96 PDF bp). Formulas, tables, plots and all supplementary pages visually inspected at rendered resolution. No overfull boxes or unresolved citations remain. Figure originals include vector PDF and 220-DPI PNG.
- Final unit/regression tests: **13 passed**. Numeric acceptance: PASS, 24 research runs, 308 rows. PDF/privacy structure acceptance: PASS, author identity present only in local report. Raw metric SHA-256 remains `2f668f644e17a7070b78d5a7453e7d02bdab05b3db97a33f2bea244c39fa6ca1`.
- Verified every recorded research-run core source hash against the original measurement commit `3a281c2`; later maintenance has not overwritten the measurement provenance.
- All source, configuration, results, figures, bibliography, README, license and anonymous PDF prepared for release. Identified local PDF, identity file, venv, data, checkpoints, logs and backup ZIP are excluded from Git.
- GPU keep-alive restoration verified with nvidia-smi: one restored process active on all four H200s.
- Status: local deliverables complete; public GitHub release awaits the user's explicit confirmation. Blackboard submission and deadline verification remain the student's external actions.
