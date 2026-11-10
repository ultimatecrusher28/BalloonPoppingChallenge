# finals config 各設定的來源表（finals/sc40_finals_lm10.yaml）

整理時間 2026-11-08 00:26（系統時間）。只用紀錄，不推測；找不到依據的寫 not found。

對象：`finals/sc40_finals_lm10.yaml`（使用者 2026-11-08 00:1x 決定 LM 10）。祖先鏈最後一個 yaml 是 LM 20 版，LM 10 版只差 `tour_deploy.polish 10`（`..._deploy_calib_lm10_final_fix.yaml`，870461b4）。檔案路徑都是團隊 repository（https://github.com/chwang7130/BalloonPoppingChallenge）裡的路徑，`/mnt/data/...` 是團隊機器上的工作目錄。
祖先鏈（每個 yaml 檔頭寫明 parent）：
`research/sc4/cfg/r8_bst.yaml`（M2FB）→ `research/sc40/cfg/m2fb_sc40.yaml` → `..._L44.yaml` → `..._L44_h10_d2.yaml` →
`..._L44_h10_d2_pc.yaml`（pass rows）→ `..._pc_cr1b18.yaml` → `..._pc_cr1b18_lc0.yaml` → `..._pc_cr1b18_lc0_bud_los.yaml` →
`m2fb_sc40_finals_fix_los6_ll16.yaml` → `m2fb_sc40_finals_ls12.yaml` → `coast_night/cfg/m2fb_sc40_finals_ls12_cut.yaml` →
`..._cut_wbt_deploy_calib.yaml` → `..._deploy_calib_lm20.yaml` → `..._lm20_final.yaml` → `..._lm20_final_fix.yaml`（870461b4）。
finals yaml 拿掉四個在現行程式沒有作用的鍵：`wall_budget_s`（.inf）、`tour_deploy` 的 `leaf`、`rank_value`、`stop_first`。

縮寫：
- WT = `/mnt/data/chwang/BalloonPoppingChallenge-wt/whole-burn-tour-force-lock-fix`
- RL = WT/docs/RESEARCH_LOG.md；ST = WT/docs/STATUS.md；RU = WT/docs/RULES.md
- R40 = WT/research/sc40；GC = R40/streams/gnc-chain.md；FN = R40/factorial/NOTES.md；CN = R40/critic/NOTES.md；
  RN = R40/guidance/rows/NOTES.md；EN = R40/endgame/NOTES.md；CL = R40/coast_night/report/critic_late.md
- AG = `/mnt/data/chwang/BalloonPoppingChallenge/.agents`；DEC = AG/decisions.md；WBT = AG/directions/whole-burn-tour-force-lock/attempts
- PA = WT/BalloonPoppingGymEnv/agents/gnc/rl/planner_agent.py；TS = R40/coast_night/critic_late/tour_shoot.py
- 「樣本內」= 設定在同一 seed 區段上選出；OOS = 樣本外。Δ 都是配對差（pops/episode）± s.e.。

## 1. planner 與 guidance（agent_kwargs 頂層）

| key(s) | value | how decided | result | n 與 seed block | commit / issue / file:line |
|---|---|---|---|---|---|
| launch_t | 44.0 | screen（gnc-chain S3） | 44 s 對 36 s +0.43 ± 0.19；40 s −0.06 ± 0.19；44 s + horizon 10 + depth 2 合計 +0.92 ± 0.17 | n 127，dev 800200–800327 | Slurm 1181、1205；RL:244-248；GC:156-170 |
| horizon、reach_kwargs.long_horizon | 10.0 / 14.0 | screen（S2）；long_horizon = horizon + 4 是設計規則，沒有單獨篩選 | h10 對 h6 +0.77 ± 0.19，h8 +0.46 ± 0.15；h12 再加 −0.02 ± 0.13 | n 127，800200–800327 | Slurm 1179；RL:237-238、246-247；GC:137-150、230-232 |
| depth | 2 | screen（S4、S7b） | depth 3 對 2 −0.05 ± 0.04，agent 時間 +42 %；h8_d2 對 h8 +0.05 ± 0.04（−25 % 時間）；Idea A replay depth 2 +0.10 ± 0.10（n 31） | n 127，800200–800327 | Slurm 1199、1206；RL:215、238-239、254-255 |
| depth_gamma | 0.6 | screen（sc1、sc4 時期）；sc40 未重調 | sc1 +0.29 ± 0.10（replicated）；sc4 +0.45 ± 0.17 | sc1 n 224（300000–300111 + 310000–310111）；sc4 n 111（600000–600111） | RL:101、133；sc40 的 0.7 arm 被使用者取消（FN:207、219） |
| beam | 16 | screen（sc4，沿用） | beam 24 / 32 / 48 各 +0.01 ± 0.05；beam 在 80–93 % 的 plan 已滿 | n 224，sc4 dev | RL:197；較早 beam 24 −0.22（RL:103） |
| branch | 12 | screen（branch 16 沒過門檻，維持 12） | sc4 branch 16 +0.08 ± 0.06（n 448）；sc40 factorial branch 16 pooled +0.075 ± 0.051，×1.10 時間，不採用 | sc40 blocks 801960–802023、802152–802279、802344–802407 | RL:197、454；FN:158-164 |
| lam | 0.15 | screen | sc40 時間成本掃描（deep critic 0.18 上）：0.10 −0.11 ± 0.19、0.05 −0.60 ± 0.17、0.20 −0.40 ± 0.20、0.25 −0.97 ± 0.23（皆對 0.15）；早期 v3 λ 0.3 → 0.15 | n 63，800456–800519 | RL:66、313-316 |
| replan_dt | 0.1 | screen（sc4）＋ 合規決定（拿掉 time-budget 模式，PR #154） | sc4 replan 0.1 對 0.25 held-out +0.46 ± 0.07；sc40 固定 0.1 s 對 budget 模式 +0.15 ± 0.12；art2 單獨 env clock 173–186 s | sc4 n 445（700672–701119）；sc40 timing Slurm 8547 | RL:193、450-451；FN:140-143、170；RU:121-127；ST:7-18 |
| variants 第 1–4 列 [2,0.8]、[3,0.3]、[4,0.3]、[3,0.3,5] | 見 yaml | quals 前（v1–v7 planner）的 guidance variants | 個別值的篩選紀錄 not found | not found | RL:58-73；PA VARIANTS |
| variants 第 5 列 [3,0.3,0,1.5,0]（aim-up 偏移） | 保留 | screen（拿掉的 arm） | 拿掉 −0.20 ± 0.12，所以保留 | n 127，801416–801543 | RL:403、500 |
| 拿掉 M2FB 的 aim-offset 列（down、side ±） | 已拿掉 | screen | 單獨拿掉 −0.06 ± 0.09（冗餘），由 line-of-sight 列取代 | n 127，801000–801127 | GC:888-889；RL:400 |
| variants 第 6–7 列 [3,0.3,10]、[3,0.3,20]（along-LOS boost） | 見 yaml | screen（sc4 round 8，M2FB） | held-out +0.36 ± 0.08 對 M2F（4 個區段都 > 0） | n 447，701120–701567 | RL:199；r8_bst.yaml:1 |
| vi_variants、vi_wr | true / 3.0 | screen（sc4 W3） | ff + vi +2.32 ± 0.17，replicated +2.00 ± 0.18；vi 單獨 +0.03；vi_wr 5 held-out +0.03；vi_wr 1.5 drop（數值 not found） | dev 600000–600111、600112–600223 | RL:150-154、467；ST:148、183 |
| los_variants 第 1–3 列 [2,0.8,0,1]、[3,0.3,0,1]、[4,0.3,0,1] | 見 yaml | screen（steering rows study，事先定規則） | pooled +0.29 ± 0.08（3.6 s.e.）；official limits +0.31 ± 0.14（n 127）；機制是 first pop 早 0.7–1.1 s | n 382，801000–801383 | GC:863-916；RL:400；使用者 11-03 採用 |
| los_variants 第 4–6 列 [1.5,0.8,0,1]、[2.5,0.8,0,1]、[2.0,0.5,0,1] | 見 yaml | screen（factorial B） | pooled +0.194 ± 0.068（2.9 s.e.），agent 時間 ×1.04；第二圈 3 列 +0.06 ± 0.11 不採用 | blocks 1 + 3：801960–802023（n 63）、802152–802279（n 127） | FN:13、126、163；RL:446-447 |
| xrows（pass rows）整組 | 見 yaml | screen ＋ replication | +0.57 ± 0.12；replicated +0.73 ± 0.12（8/8 hours > 0）；official limits +0.59 ± 0.28（n 32） | n 127 800200–800327；n 128 800328–800455 | RL:259-264；RN:203-213；merge 5790849a |
| xrows.variants [[0,0.1,1.0],[0,0.3,1.0]]（m 0 = 點質量最短時間、slack 0.1 / 0.3 s、beta 1） | 見 yaml | 離線 probe（第一版失敗後重設計） | 第一版 T*(0.9) 閉迴路 −0.16 ± 0.14（n 32）；兩段測試 beta 1（pa）−0.12 s、beta 0.5（pb）−0.04 s；slack 0 / 0.2 較差（waypoint 模式，+0.30 / +0.29 s，n 54） | 75 個錄製 plan 狀態（800200–800231 的飛行） | RN:54-77、113；planner.py enable_x |
| xrows.rate_cap、xrows.lead | 1.5 rad/s / 0.15 s | 離線 probe（單一設定 pc 對 pa） | 兩段 Δ median −0.40 s（pa −0.12 s），sweep 有解 96–98 %；數值掃描 not found | 75 個 plan 狀態 | RN:70-77 |
| xrows.rp | 0.6 m | screen | 中心 pass（rp 0.05，pc0）→ 0.6 m 偏心 +0.75 ± 0.19 | n 32，800200–800231 | RN:188-191、208 |
| xrows.k2 | 2 | 設計；反向 screen | k2 4 −0.09 ± 0.11；拿掉 next-balloon coupling −0.45 ± 0.14 | n 127，801416–801543 | RN:18；RL:403-405、501 |
| xrows.aim | pmp | 設計（Pontryagin 時間最佳 leg 加 look-ahead exit） | 單獨篩選 not found | not found | RN:8-16、115-120 |
| xrows.stick_fb | true | 設計修正（第一版失敗後加 sticky fallback） | 單獨篩選 not found | not found | RN:55、67-68 |
| reach_kwargs.reach_beta | 0.5 | screen | sc4 held-out +0.19 ± 0.07；sc40 0.25 pooled +0.10 ± 0.10（n 255），0 −0.82 ± 0.29（n 39），0.75 −0.44 ± 0.13（n 81）；0.5 為此族最佳 | sc4 n 445（700672–701119）；sc40 801672–801927、802024–802151 | R40/../sc4/streams/momentum.md:42；RL:418-420、443、502-503 |
| reach_kwargs.reach_mode、reach_k、reach_tau0 | soft / 3 / 2.0 | 設計（M2） | soft β 0.5 dev +0.53 ± 0.14；k 3 與 tau0 2.0 的掃描 not found | sc4 dev 600112–600223 | WT/research/sc4/streams/momentum.md:34；WT/research/sc4/cfg/m2_soft_b05.yaml:1 |
| critic_path | critic_r1_deep.npz | 訓練（deep-plan target，round 1） | val R² 0.813；leaf 選到最佳 66 %（舊 critic 56 %）；round 3 critic pooled −0.03 ± 0.09，未換 | 訓練 seed 17–144（sc40 cache），26 局依 seed 留出 | CN:23-50、66-81；RL:362-363 |
| critic_beta | 0.18 | derivation ＋ screen | 新 critic within-plan spread 是舊的 3.4 倍，0.6 / 3.4 ≈ 0.18；權重曲線 0.10 +0.03、0.18 +0.19 ± 0.13（n 127）、0.36 −0.27 ± 0.21（n 63）；fresh blocks（critic + lam_coast 0）對 pass rows +0.12 ± 0.10（n 256） | 800456–800583；robustness 800584–800711 | CN:117-123、143-144、182-184；RL:297-298、377-379 |
| lam_coast | 0.0 | screen | 對 cr1b18 +0.13 ± 0.04（coast +0.23 ± 0.05，無計算成本）；critic + lc0 對 pass rows +0.32 ± 0.12（tuning）、+0.18 ± 0.13（shared block） | n 127 800456–800583；n 128 800328–800455 | CN:203-205、244-246；RL:328-330 |
| lam_late、lam_late_t | 0.1 / 16.0 | screen（factorial D）＋ 確認 | pooled +0.094 ± 0.037（2.5 s.e.），×1.02 時間；與 3 列 LOS 合併的確認 +0.219 ± 0.117 | blocks 2/3/5：802024–802151、802152–802279、802344–802407；確認 802408–802535（n 128） | FN:44、164、168-170、184-186；RL:452-458 |
| late_search（t、depth、beam、branch） | 12.0 / 3 / 32 / 16 | screen（隊友 endgame E2 + E3，事先定規則）＋ 使用者核准 | 12 s 起 pooled +0.20 ± 0.06（3.4 s.e.），≤ 6 比例 10.7 % → 6.3 %；16 s 起 +0.06 ± 0.06、20 s 起 +0.02 ± 0.05 不採用；16 s 時 depth 3 alone +0.04 ± 0.07、beam + branch alone −0.12 ± 0.09 | n 380 paired，802856–803047 | EN:79-106；DEC:266；commit b3f4dcdb；m2fb_sc40_finals_ls12.yaml:1-2 |
| predictor | true | screen（sc4 P1） | 對 dg06 +1.26 ± 0.18；held-out 6.43 ± 0.09（n 223） | n 111，600000–600111 | RL:140-142 |
| pred_sim_wind、pred_sim_wind_zlo | true / 30.0 | screen（#7） | 對 M2FB +0.07 ± 0.11（sc4 時期同類做法 −0.23 ± 0.15）；30 m 是使用者 2026-10-30 定的風量測帶下緣（#4，chwang-wind-estimation@5576ad42 的 research/sc40/streams/wind-estimation.md：seeds 0–49 × 8 hours 開迴路量測，20–30 m 那一格佔量測帶均方誤差 91 %，之後量測帶定為 30 m … z_top − 150 m） | n 256，held-out 900000–900031 × 8 hours（紀錄如此） | commit eae7ab02（#7）；RL:142；m2fb_sc40.yaml:1-2 |
| rocketnav | mekf（model-aided 19-state filter） | measurement（導航誤差目標）＋ benchmark | 閉迴路 64 seed:hour 全部達標（boost 位置 0.094 m、nose 1.23°、rate 2.95°/s）；R1 沒有既有因果方法勝過它；truth_nav oracle 只多 +0.13 ± 0.15；對舊 MEKF 的 pops Δ not found | 64 seed:hour；oracle n 127，800200–800327 | commit d3f22c7f（#6）、34ae7ef4；RL:222-224、233-234 |
| loop_kwargs.kp_rate | 10.0 | screen（sc3，pre-W3） | rate gain 25 / 15 / 12 / 10 / 8 → 2.50 / 4.79 / 5.14 / 6.07 / 5.79 pops；sc40 重調 not found | 14 個 sc3 dev seed | RL:76-77（RL:155 註明 pre-W3 調參待重驗） |
| loop_kwargs.ff | true | screen（sc4 W3） | act_inv +1.86，+ ff +1.99，ff + vi +2.32 ± 0.17；replicated +2.00 ± 0.18 | dev 600000–600111；600112–600223（n 未寫明） | RL:150-154 |
| sim_burnout_cut | true | derivation（env 燃盡時刻）＋ measurement ＋ screen；master 以模型修正保留 | planner coast 殘差 3 s 1.14 → 0.32 m；fork 篩選 Δtotal +0.00 ± 0.06 | n 128 對，64 parent 802408–802471 | CL:28-31；DEC:394；m2fb_sc40_finals_ls12_cut.yaml:1；使用者 23:2x 納入（DEC:525） |
| sim_terms_t | 24.0（sim_axial、cm_kin、grav_model 從 24 s 打開） | measurement ＋ screen ＋ 使用者決定（22:1x 採用、22:2x 選 24 s 打開） | 26–30 s 軸向殘差 −0.134 → +0.011 m/s²；fork Δ total −0.031 ± 0.058（null），≤ 6 局 13 → 9 | n 128 對，64 parent 802408–802471 | AG/directions/burn-thrust-model/status.md:14-30、82-90、107-110；DEC:505-506；commit d88bd422、549ede9b |
| shoot_reshoot_late | (4.0, 0.1, 0.05, 8) | screen（tourlate4）；dt、tmin 沿用 shoot_fine_t，iters 沿用 shoot_conv_t 的 8 步 | 32 parent Δtotal +0.05 ± 0.06、coast +0.13 ± 0.07，鎖定後兩顆都中 9/11（1 s 版 5/11）；64 parent Δtotal +0.023 ± 0.030、coast +0.086 ± 0.041，≤ 6 沒增加；t_left 4 以外的數值掃描 not found | 802408–802471，64 / 128 對 | commit 18eae36a、ca939bb3；DEC:370、399-400、406；CL:338-346；AG/directions/tourlate4-screen-64/status.md |

## 2. tour_deploy（設定值）

| key(s) | value | how decided | result | n 與 seed block | commit / issue / file:line |
|---|---|---|---|---|---|
| tour_deploy 整體（mode classes、value calib、24 s ＋ 26 s） | 見 yaml | screen ＋ OOS | 樣本內 calib +0.312 ± 0.061；OOS calib（LM 10，只 24 s）+0.234 ± 0.064；OOS 24 + 26：LM 10 +0.289 ± 0.067、LM 20 +0.234 ± 0.074 | 樣本內 64 parent 802408–802471（128 對）；OOS 64 parent 802472–802535 | WBT/fork-screen.md:77-106；WBT/wall-cuts.md:49-58；DEC:439、464、500 |
| value calib（對 force） | calib | screen | 樣本內 calib +0.312 ± 0.061 對 force +0.242 ± 0.079（force 有 2 對進入 ≤ 6）；OOS force K 60 +0.211 ± 0.086，802485 7 → 6 觸發停止規則 | 128 對 ×2 區段 | WBT/fork-screen.md:77-106；DEC:464 |
| calib 常數：calib_coast、calib_burn、calib_end_in、calib_end_after（程式預設） | 0.912 / 0.06 / 0.18 / 0.30 | measurement（零成本校準）＋ 使用者決定（14:2x 採 master 提案） | coast 命中 31/34 = 0.912（17 個 tourlate4 鎖定）；燃燒段多 +0.06；base 偏差：leaf 在燃燒內結束 −0.18，燃盡後結束時 npop 單用 −0.30（74 點）；重算命中 0.854 時 master 維持 1.884 | 30 個 learned T 24 狀態；tourlate4 64 parent | commit a42d0112（R40/coast_night/report/gate/lock_value_calibration.md）；PA:83-84；DEC:416、428 |
| t | 24.0 | 離線 measurement ＋ 使用者決定 | 不限 wall 求解推算 calib 鎖定：T 24 20/32、T 26 17/30、T 28 5/29；20 s 決策 parked（20 + 24 + 26 對 24 + 26 +0.062 ± 0.077） | 802408–802439；T 20 篩選 802408–802471 | AG/directions/two-coast-tour-nolimit-solve/attempts/t26-class-solve.md 表；DEC:416、485、491 |
| retry_t | [26.0] | screen ＋ 使用者決定（20:5x 定 24 + 26） | 26 s 第二次機會 +0.070 ± 0.041（樣本內，LM 20）；OOS 增量 +0.047 ± 0.027（LM 20）、+0.055 ± 0.028（LM 10） | 64 parent 802408–802471；OOS 64 parent 802472–802535 | DEC:485、491、500；WBT/wall-cuts.md:5-9 |
| seq_depth | 已移除（使用者 2026-11-08 01:0x） | 加深層只寫 seq_log["deep"]，classes 模式不讀；程式與 yaml 一起拿掉，802414:18、802446:18 對 870461b4 逐位元相同 | 原為 8：加深 leaf 沒有更準（燃燒段集合相同 0.033，base 0.109），plan wall p50 1.88 → 2.65 s | n 64，block 6 B0 | CL:228-242；CL:333-336 |
| mode、n_cls | classes / 10 | 設計 ＋ 離線 measurement | 四類各取前 10；T 24 有決策的 30 局中 27 局至少一類有通過解；nburn、seeds 4、K 36 下最多只用到前 9 個候選 | 802408–802439 | AG/directions/two-coast-tour-nolimit-solve/attempts/t24-class-solve.md；WBT/wall-cuts.md:13-15；commit f454d6bb |
| pass_coast | true | 設計（目標：燃燒段加 2 顆 coast，兩顆 coast 都要 ≤ coast_hit） | 單獨篩選 not found | not found | WBT/impl.md；AG/directions/whole-burn-tour-force-lock/status.md:6-7 |
| gens、pop、es_dt | 40 / 128 / 0.02 | screen（solver-cost grid） | 參考設定 4 seed 內通過 46/116；沒有便宜 2 倍又保住大部分通過解的設定（128 × 10 代 28/116、wall 0.45×）；es_dt 0.01 55/116 但 wall 1.67×，每 1000 s 通過數較低；pop 64 37/116 | 116 候選，61 parent（802408–802471），fresh seed 4–7 | WBT/solver-cost.md:39-69；R40/coast_night/whole_burn/solver_cost_grid_sum.md；commit db764d46 |
| sigma0、sigma0b | 0.15 / null | screen（solver-cost） | 每 seed 通過率 sigma0 0.15 0.241 對 0.3 0.224；sigma0b null 沒有損失 | 116 候選 | WBT/solver-cost.md:59-61；DEC:443 |
| polish | 10（決賽；先前 LM 20） | 求解器 screen ＋ 使用者決定 | LM 20 4 seed 內通過 60/116 對 LM 10 46/116，wall 1.24×（使用者 18:5x 改 LM 20）；閉迴路 OOS LM 20 − LM 10 −0.047 ± 0.048（24 s）、−0.055 ± 0.046（24 + 26）；修正開的重新評估 lm10 − lm20 +0.055 ± 0.034（null），決策 p90 120 對 146 s；**使用者 2026-11-08 00:1x 決賽用 LM 10，submission 只留 lm10 yaml** | OOS 64 parent 802472–802535；重新評估 64 parent 802408–802471 | WBT/solver-cost.md:70-116；WBT/fork-screen.md:107-121；WBT/wall-cuts.md:101-104；DEC:455、463、483、2026-11-08 00:1x 列 |
| seeds | 4 | counterfactual ＋ 使用者決定（17:0x 定暫定值） | P(seed s 通過 ｜ 前面沒過) 30/116、9/86、6/77、5/71，支持每候選 2–4 seed；s ≤ 4 加 sigma0b null 等於原 8 seed 排程的前 s 個 | 116 候選 | WBT/solver-cost.md:63-64；WBT/deployable-schedule.md:5-15 |
| order、sched | nburn / cand | counterfactual（零成本，12 個一批） | nburn 逐候選 3 批每 parent +0.338，比 seed-major 多 +0.051 ± 0.026；force 在 K ≥ 6 時 nburn 多 +0.03 到 +0.10 | 64 parent 802408–802471（樣本內） | DEC:424；AG/directions/tour-lock-own-coast-gate/status.md:28-29（batch12_cf@74b776dc） |
| seed_budget | 36 | counterfactual ＋ OOS | calib K 12 / 24 / 36：+0.268 / +0.311 / +0.331 ± 0.062，K ≥ 36 決策不變；OOS 每個鎖定 ≤ 33 次求解，K 36 = K 60 | 樣本內 64 parent；OOS 64 parent 802472–802535 | DEC:447；WBT/fork-screen.md:96-106 |
| wall_max | null | 合規（PR #154：決策不讀 agent 時鐘） | 假時鐘測試 3/3 SHA、pop、決策相同 | 802410:6、802423:21、802447:21 | WBT/deployable-schedule.md:102-107（6244）；RU:121-127 |
| near_t、burn_max_near | 1.0 s / 1.4 m | measurement（derivation）；near_t 由 master 08:4x 選 b | s802441 第一顆目標在 T 後 0.4 s 經過，鎖定到 env 的漂移 +0.08 m（取 0.1 m），1.5 − 0.1 = 1.4 m | 1 局 2 個 stream（l44 錄製） | CL:218、252、257；commit 46ded316；PA:39-40 |
| ntop | 300 | 設計（沿用不限 wall 求解的 rank_ntop 300：每類前 300 名精確） | 預設 20 在 266240 候選檢查上前 20 名不變 | 單一檢查 | R40/coast_night/two_coast/nolimit_classes.py:10、290；commit 46ded316；WBT/impl.md:15 |
| parallel、workers、pin_cores | true / 12 / true | compute cap（使用者 15:5x，每局最多 12 實體核心）＋ 身分 ＋ 計時 | 串列對平行、串列對綁核 18/18 決策與 SHA 相同；綁核每批 1.07× 單 seed p90（未綁 1.47×）；art2 單獨 env clock max 436.9 s（LM 20）；換算最壞 512–532 s | 3 個狀態；art2 12906 共 6 局 | chwang-rules/agent-packages@e233c09a docs/RULES.md:140；DEC:423、433、517；WBT/deployable-schedule.md:67-100；WBT/wall-cuts.md:68-100 |
| rank_lazy、queue | true / true | 使用者決定（20:5x，wall 減法 1 與 3）＋ 身分 | 開與不開 9/9 決策、SHA、pop 相同；rank_lazy 在 24 s 省約 4 s；queue 在 36 seed 都沒鎖時不省時間 | 802410:6、802423:21、802447:21 | WBT/wall-cuts.md:5-32、68-80；commit c123daca、f9d4c427 |

## 3. tour 程式預設值（TOUR_DEPLOY，PA:78-85）

| key(s) | value | how decided | result | n 與 seed block | commit / issue / file:line |
|---|---|---|---|---|---|
| K（程序段數） | 8 | 離線 screen | K 8 比 12 / 16 / 24 好；CMA-ES + LM 的 2/84 通過都是 K 8 | 6 局 l44 錄製 | CL:144-163 |
| h、smax | 0.04 / 0.15 rad | derivation ＋ 離線比較 | 路徑對角度不平滑（GN 步 0.5–4 rad 成本升 2–45 倍），所以加單步上限 0.15 rad；h 比較 0.01 與 0.04，混合求解器用 0.04；smax 數值掃描 not found | 指定序列 2 局 | CL:153、159-161；WT/BalloonPoppingGymEnv/agents/gnc/rl/tour.py:5-6 |
| win、win_sub | 1.5 / 2.5 s | 設計 | 燃燒段目標 pass 窗口為 pop 時刻 ±1.5 s，候選 ±2.5 s；掃描 not found | not found | CL:137；TS:43 |
| coast_t | 3.5 s | 設計 | coast 窗口由燃盡後 6 s 改為 3.5 s；改的理由 not found | not found | CL:137、164；TS:42 |
| burn_max | 0.8 m | measurement（derivation） | 1.5 m 半徑減去鎖定到 env 的燃燒段漂移（≤ 0.7 m） | s802441 2 個 stream | CL:218、252；PA:36-37 |
| coast_hit | 1.2 m | derivation | planner 命中半徑 0.8 × 1.5 m | — | CL:137；TS:40 |
| coast_stop | 0.96 m | derivation | 0.8 × 1.2 m（求解在接受門檻的 0.8 倍停止，同 fork A 的 tour_shoot） | — | TS:87、133、445；PA:38 |
| w_burn | 2.0 | 設計（沿用 tour_shoot --wb 預設） | 掃描 not found | not found | TS:322；PA:38 |
| nseg | 40 | 設計（最小能量排序的常數推力段數） | 掃描 not found | not found | CL:179-181；tour.py:425 |
| rsub、rmax | 1.0 / 1.1 | derivation ＋ 離線 measurement | 能量比 ≤ 1 是此模型下可行的必要條件；T 28 放寬到 2.0：有候選的局 15 → 26，有通過解只 9 → 11，比值 > 1.1 的候選只解出 11/204 | T 28，29 局 | CL:179-185；AG/directions/two-coast-tour-nolimit-solve/attempts/t28-rmax2.md |

## 4. 找不到來源或需注意的地方
- 找不到單獨證據：variants 第 1–4 列的數值、xrows.aim 與 stick_fb 的單獨效果、xrows.rate_cap / lead 的數值掃描、reach_k 與 reach_tau0、對舊 MEKF 的 pops 差、pass_coast 單獨效果、tour 預設的 smax、win、win_sub、coast_t（6 → 3.5 s 的理由）、w_burn、nseg 的掃描。
- 只在 sc1 / sc3 / sc4 驗證、sc40 沒有重調的：depth_gamma、beam、kp_rate、ff、vi_variants / vi_wr、predictor、reach_mode / k / tau0。RL:155 註明 pre-W3 的調參（kp_rate 等）應重驗。
- 樣本內選出、只有部分 OOS：tour_deploy 的 order、seeds、seed_budget 在 802408–802471 上選，OOS 802472–802535 確認 calib（K 36 = K 60）。
- pred_sim_wind 的唯一 pop 數字用的是 held-out 900000–900031（commit eae7ab02 訊息原文）。
- polish：決賽用 LM 10（使用者 2026-11-08 00:1x）。依據：guidance-ocp-gap lm-reeval@980db4bb report/lmre_main.md，lm10 − lm20 600 s 內 pops +0.055 ± 0.034（n 64，樣本內，null），決策時間 p90 120 對 146 s；先前 OOS 802472–802535（修正未開）lm20 − lm10 −0.055 ± 0.046（learned t20 638f68ac）。
