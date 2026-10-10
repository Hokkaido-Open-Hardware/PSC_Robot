# PSC_Robot

PSC_Robot is a two-wheeled balancing robot simulation project built with MuJoCo and a PySide6 desktop interface. It supports LQR, neural network, and reinforcement learning controllers, with tools for training and evaluation.

## Features

- **LQR:** The default controller for balancing and movement.
- **Neural network:** A supervised controller trained to imitate LQR, with FP32 and INT8 inference options.
- **Reinforcement learning:** PPO training for balance and velocity tracking.
- **Interactive controls:** Adjust speed and yaw manually, reset the robot, or use AUTO mode to run predefined command patterns.

## Repository layout

- `balance-robot-mujoco-sim/`: Simulator, training tools, and evaluation scripts.
- `balance-robot-mujoco-sim/models/`: Saved models and evaluation results.
- `balance-robot-mujoco-sim/tests/`: Automated tests and GUI checks.
- `external/PSC-ONE/`: PSC-ONE hardware and software submodule.

## Getting started

Install Conda and use a desktop environment with OpenGL support. From the repository root, create the environment and launch the simulator:

```bash
cd balance-robot-mujoco-sim
conda env create -f conda-environment.yaml
conda activate robot-mujoco-simulation
python src/simulation/simulate_robot.py
```

The simulator starts in LQR / MANUAL mode. Use the Speed and Yaw sliders to control movement. Reset restores the robot while retaining the selected controller.

If you need the PSC-ONE sources, initialize the submodule from the repository root:

```bash
git submodule update --init --recursive
```

## Cleaning Python caches

From the repository root, run:

```bash
make clean
```

This removes `__pycache__`, `.pyc` / `.pyo` files, and pytest, mypy, and ruff
caches. Virtual environments, `external/`, `build/`, and saved models are retained.

## Optional controllers

Install the dependencies for the features you need from `balance-robot-mujoco-sim/`, with the Conda environment active:

```bash
python -m pip install -r requirements-nn.txt    # Supervised NN training
python -m pip install -r requirements-rl.txt    # PPO training and control
python -m pip install -r requirements-int8.txt  # INT8 inference
```

LQR and exported FP32 neural network inference only require the base environment. Use the GUI training panels to train models and view progress. For a saved RL model, use **Load RL** before selecting **Reinforcement Learning**. AUTO mode runs command patterns for evaluation; it does not update model weights.

## Documentation

See the simulator documentation in [English](balance-robot-mujoco-sim/README.md) or [Japanese](balance-robot-mujoco-sim/README_JP.md) for training commands, model formats, and evaluation details. INT8 results are documented in the [validation report](balance-robot-mujoco-sim/docs/int8-validation.md).

## License

[MIT](LICENSE). See the included projects for their respective license terms.

## PSC-ONE 統合シミュレーション

cocotbを起点にMuJoCo子プロセスを起動し、PSC-OSがI2Cで受け取った状態から
計算した車輪速度でロボットを動かします。既存のLQR・NN・RL単独起動に加え、
GUI右側の **PSC-ONE integration** から連携を起動・停止できます。
PSC-OSのブート完了後、同じウィンドウ内のロボットをPSC-OS制御へ切り替えます。
連携終了時は元のコントローラーへ戻してリセットします。
連携ウィンドウではSpeed・Yaw・MANUAL/AUTO・Resetを操作できます。
ローカル制御への切替・学習操作は無効です。終了後は親ウィンドウの単独動作を続けられます。
SpeedはI2CのINT8形式に合わせて−0.512〜+0.508 m/sです。Yawはロボット側で
PSC-OSの出力に加算します。AUTOは通常GUIと同じシナリオを物理時刻で進めます。
最初のStoppedシナリオは8秒の物理時間、目標速度0です。AUTO中は手動スライダーが無効です。
AUTO表示の経過時間は `simulation time`（シミュレーション時間）です。
Stopped中は停止指令の保持と次のGentle Forwardを表示します。PSC-ONE連携が低速な場合、
最初の8秒に実時間で数分かかることがあります。
通信待ち中のReset・モード変更では、その待機中の応答を破棄して新しい状態から再開します。

### 配置と関連コード

`external/PSC-ONE` の既存submodule（`a28fcadf14b825c6d8c2a36b1407b0509e3e4711`、
`v0.9.9.1`）を利用します。`.gitmodules` のURLは
`https://github.com/Hokkaido-Open-Hardware/PSC-ONE.git` です。
調査時点でPSC_Robotと配置済みPSC-ONEの双方に `AGENTS.md` はありません。
PSC-ONE側のソフトウェア変更もsubmodule内の変更として管理してください。
親リポジトリの変更だけを保存しても、そのソフトウェア変更は含まれません。

PSC-OSはローカルの `~/Program/PSC_RV32I/PSC-ONE/software/os/` をコピー元として更新します。
リポジトリルートで次を実行してください。コピー元の未コミットの変更も含みます。

```bash
make sync-psc-os
# コピー元を変更する場合（PSC-ONEディレクトリを指定）:
make sync-psc-os PSC_ONE_SOURCE=/path/to/PSC-ONE
```

`external/PSC-ONE/software/os/` の同名ファイルを内容比較して上書きします。
起動用 `.S` とテストソースもコピーし、ビルド生成物・Pythonキャッシュ・Git管理情報は除外します。
コピー先だけにあるファイルは削除しません。更新はこのコマンドで明示的に行います。
submoduleの除外設定により `.S` や `tests/` は通常の `git status` に表示されないため、
コミットに含める場合は対象を明示して `git add -f` してください。

| ファイル | 役割 |
| --- | --- |
| `external/PSC-ONE/hardware/sim/cocotb_tb/mmio/psc_rv32_i2c.py` | 既存I2C単体テストと再利用するPythonマスター |
| `external/PSC-ONE/software/os/src/api/i2c_api.{c,h}` | 既存MMIO API。入力4バイト、出力1バイト、W1C状態、8bit連番 |
| `external/PSC-ONE/software/os/src/api/robot/robot_control.{c,h}` | 追加した固定小数点LQRサービス。CPUが計算し、NPUは使用しない |
| `external/PSC-ONE/software/os/src/kernel/kernel.c` | `PSC_ROBOT_SIM` 専用ビルドで初期化後にサービスを起動 |
| `integration/psc_one/Makefile` | 既存PSC-OS/SoCソース一覧を使い、専用領域へビルド |
| `integration/psc_one/cosim.py` | 起動確認、I2C変換、時間同期、ロボット子プロセス管理 |
| `integration/psc_one/robot_cosim.sv` | 製品RTLを変更しないテスト専用の信号アダプター |
| `integration/psc_one/boot.S` | 統合シミュレーションで使用するテスト専用起動コード |
| `integration/psc_one/run.py` | ビルド・実行の監督、タイムアウト、プロセスグループ回収 |
| `balance-robot-mujoco-sim/src/simulation/psc_protocol.py` | サイズ制限・バージョン・連番を持つUnixソケットIPC |
| `balance-robot-mujoco-sim/src/simulation/psc_worker.py` | MuJoCo状態取得、返信した車輪速度の適用、物理ステップ |
| `balance-robot-mujoco-sim/src/simulation/psc_panel.py` | GUIの起動・停止と通信専用スレッド |

コピー元の通常OSには `robot_control` シェルコマンドによるI2C受信表示があります。統合シミュレーションのサービスは
専用ビルドのカーネル内でポーリング実行し、シェルやユーザープロセスを起動しません。
通常OSビルドの起動経路は変更していません。

### 現在のハードウェア制約

配置済みソースには外部I2C経路の配線問題があります。

- `hardware/rtl/tang20k/top/PSC_ONE_Chip.v` はSoCのSCLを
  `lv_drive_low[2]` に接続しており、外部ピンの入力を使用していません。
- `hardware/rtl/soc/soc_top/PSC_ONE_SOC.v` のSDAが `output` 宣言です。
  スレーブの入力にも使われますが、上位側でも `lv_drive_low[3]` に接続されています。
- `PSC_ONE_Chip_sim.v` はLV1/LV2に接続する一方、SoC側はインデックス2/3を使います。

このため、専用アダプターは既存 `u_i2c.scl_i/sda_i` のみをforceし、SDAを
`~(master_sda_low | slave_sda_low)` で解決します。CPU、キャッシュ、MMIO、I2Cスレーブの
処理は実RTLを通り、PythonがMMIO入力や制御出力を注入することはありません。
**これは基板・チップ外部ピン経由の動作確認にはなりません。** 製品RTLは変更していません。

実機経路の修正案は、基板の実際のLVピン割当を確認してSCLを `lv_input` に接続し、
SDAをSoCの `inout`、または入力とdrive-low出力に分離し、ピン境界で一度だけ
open-drainを解決することです。シミュレーション上位のLV割当も同じピンに揃える必要があります。
CPU・NPU・I2Cレジスタ仕様の変更は不要です。この製品RTL修正は今回適用していません。

既存OS Makefileが要求する `src/boot/psc_mbios.S` はローカルコピー元から補完しました。
`boot.S` はS-modeの既存カーネルentryへ移る専用fixtureで、SBI・割込み・warm restart・
通常シェルは統合シミュレーションの検証対象外です。

### プロトコルと同期

ロボットプロセスが論理的なI2Cマスターです。Unixソケットの状態要求をcocotbが代理して
スレーブアドレス `0x42` のSCL/SDA信号に変換します。レジスタ0から4バイトを書き、
レジスタ4のOUTPUT_READYを待ち、8の出力と9のSEQを読みます。
起動時はPSC-OSが `OUTPUT_READY=1, OUTPUT=0, SEQ=0` を設定して受付準備を通知し、
既知入力 `[10,0,0,0]` に対する出力 `-10` を確認してからロボットを起動します。

| 入力順 | 内容 | 1 LSB |
| --- | --- | --- |
| 0 | 既存LQRと同じ符号のpitch | 0.004 rad |
| 1 | 既存LQRと同じフィルターのpitch rate | 0.04 rad/s |
| 2 | 既存LQRと同じフィルターの平均車輪角速度 | 0.5 rad/s |
| 3 | 目標並進速度（MANUAL/AUTO指令） | 0.004 m/s |
| 出力 | 共通車輪角速度 | 0.25 rad/s |

すべてzero point 0の符号付きINT8で、最近傍丸め（ちょうど半分はゼロから遠い方）と
`[-128,127]` 飽和を適用します。出力範囲は `[-32,31.75] rad/s` です。
このwire形式は既存NNの量子化形式とは異なります。
PSC-OSは既存LQR式に対応したQ16係数で計算し、左車輪に `-output + yaw`、右に `output + yaw` を適用します。
Pythonはフィルター・量子化・モーター符号変換を担当し、ローカルLQRの制御計算は呼びません。

制御周期は5ms、MuJoCoの物理刻みは25µs（200ステップ）です。
1周期につき状態を1回送信し、返信までは物理を進めません。
cocotbはI2C要求・OS処理・返信を5msのRTL時間内で行います。
`--timing strict`（ヘッドレス既定）では残りのRTL時間を進めてから返信します。
`--timing interactive`（GUI既定）では応答後の空き時間を省き、起動確認後のI2C信号の
各待ち時間を5000nsから1250nsに短縮します。MuJoCoの5ms刻み・全要求の実CPU計算・
実I2C転送は維持しますが、RTL時刻と物理時刻は一致しません。厳密な時間検証にはstrictを使います。
両モードともクロック生成はcocotbのGPI実装を使用し、エッジごとのPython実行を省きます。
その出力を同じ論理周期のMuJoCoの200ステップに一定値として適用します。
これは離散時間の同期モデルであり、I2C遅延を物理系内の追加遅延として再現する方式ではありません。
OSブート時間はロボット時間に含めません。

IPC待ちの間はRTL時間も止まります。壁時計の速さやGUIの描画速度による追いつき・
周期の間引きはありません。GUI下部には実時間に対する進行倍率も表示します。IPCの非負整数のPython連番と論理時刻、I2Cの8bit連番を照合し、
連番周回はI2C側のみmod 256で扱います。未処理入力がある間の次フレーム送信は禁止です。

起動応答はRTL時間2秒、各制御応答はRTL時間5ms、子プロセス接続は壁時計60秒、
IPC待ちは壁時計120秒で打ち切ります。全体はCLIの `--timeout` で制限します。
NACK、overrun、連番不一致、NaN/Inf、切断は連携をエラーで終了します。
通常GUIでは連携終了後、ロボットをリセットして元のローカル制御へ戻します。
GUIのIPC待ちは描画ロックの外で行い、終了時にソケットshutdownで解除します。
監督プロセスはSIGINT/SIGTERMでmake・シミュレーター・MuJoCoのプロセスグループを
TERM→KILLで回収し、Linuxではsubreaperで孤児化した孫プロセスもwaitします。
同時起動はファイルロックで拒否し、通常のPSC-ONEビルド出力をcleanしません。

### 実行

Linux、MuJoCo/PySide6/SciPy/NumPyを含む既存ロボット環境、cocotb 2.0、Verilator、
Clang、LLVM objcopy、RV32IM用libgcc、make、hexdumpが必要です。
I2C単体テストはIcarusも使用します。cocotbとMuJoCoは別Python環境でも構いません。

```bash
# 同じ環境にcocotbも導入する場合
myenv/bin/python -m pip install -r integration/psc_one/requirements.txt

# ヘッドレス統合実行。全体の壁時計タイムアウトにはビルド時間も含む
myenv/bin/python integration/psc_one/run.py --steps 20 --timeout 1800

# GUI用の高速モードをヘッドレスで検証
myenv/bin/python integration/psc_one/run.py --steps 20 --timing interactive

# cocotbが別環境の場合
myenv/bin/python integration/psc_one/run.py --steps 20 \
  --sim-python /path/to/cocotb-env/bin/python

# 通常GUIから右側パネルのcocotb Pythonを設定してStart / Stop
myenv/bin/python balance-robot-mujoco-sim/src/simulation/simulate_robot.py

# cocotb起点で連携GUIを直接開く
myenv/bin/python integration/psc_one/run.py --gui --steps 1000000 \
  --timeout 86400 --sim-python /path/to/cocotb-env/bin/python
```

ビルド、実行ログ、cocotb XML結果は `build/psc-one/` に出力されます。
`--out build/別名` で検証用の独立した出力先を指定できます。
ビルド・実行ログは起動元のターミナルにも随時表示されます（通常GUIからのStartも同様）。
PSC-OSのUART送信はシリアル信号から復元し、`[PSC-OS UART]` を付けて表示します。
これらの出力は `build/psc-one/run.log` にも保存されます。
通常GUIのPSC-ONE integration欄には、CPUのPCがカーネルの`memset`範囲内にある間、
`PSC-OS state: memset 実行中` を表示します。アドレス範囲は実行時のビルド結果から取得し、
RTL時間10µsごとに確認します（それより短い呼び出しは表示されない場合があります）。
実機速度での実行は想定していません。最初の確認ではブート込み3周期に約161秒かかりました。
通常GUIのStopは連携を終了し、同じウィンドウでローカル制御を再開します。
再起動時にはOS・ロボットの状態を初期化します。

### PSC-ONE制御移行後の10秒録画

右側の **PSC-ONE recording** にある **Record 10 s after PSC-ONE handover** を
有効にしてから連携を開始すると、制御移行直後からシミュレーション時間で10秒間の
ロボットの動きを記録し、リポジトリ直下の `recordings/psc-one-日時.mp4` に保存します。
初期設定は有効です。直接 `--gui` で起動した場合も録画します。
録画にはPATH上の `ffmpeg`（libx264対応）が必要です。未導入の場合は録画欄に表示します。

MP4は640×480、20fps、通常速度で10秒の動画です。ロボットの3D表示だけを保存し、
カメラは保存開始時の画角を使用します。PSC-OSのビルド・ブート・通信待ちは動画に含めません。
RTLシミュレーションの進行速度に応じて、記録完了までの実時間は10秒より長くなります。
物理状態を一定間隔で保持してから動画に変換するため、画面の描画速度によるコマ落ちはありません。
10秒経過後もPSC-ONEによる制御は継続します。

途中でStop・切断・ウィンドウ終了が発生した場合は、取得済みの動きを
`-partial.mp4` として保存します。最初の制御ステップ前に停止した場合は保存しません。
ResetやAUTOによるリセットは動画にそのまま含み、録画時間は実行済み物理ステップの累計です。
進捗・保存先・録画エラーは録画欄に表示し、録画の失敗で制御を停止させることはありません。

通常GUI・PSC-ONE連携GUIの右側上部に `PSC-ONE I2C dump` を表示します。
初期ウィンドウサイズは1536×1080（従来の縦横1.2倍）です。
実際のI2C通信をRTL時刻、7bitアドレス、レジスタ、送受信バイト（16進）、
START/STOP、ACK/NACK付きで確認できます。TXはmaster→PSC-ONE、RXは逆方向です。
読み出し最終バイトのNACKは正常な終了通知です。起動時のstatusポーリングも含めて
`build/psc-one/i2c.log` に保存し、GUIには最新2000行を表示します。
`Auto-scroll` を外すと過去の通信を確認でき、`Clear` は表示だけを消去します。
再起動するとログと表示が新しい実行に切り替わります。

```bash
# IPC・ファームウェア計算・MuJoCo子プロセス・終了処理
myenv/bin/python -m unittest discover -s balance-robot-mujoco-sim/tests \
  -p test_psc_integration.py -v

# 元からあるI2C単体テスト。cocotbの仮想環境を有効にして実行
cd external/PSC-ONE/hardware/sim
make -f ../../../../integration/psc_one/Makefile i2c-tests
```

### 検証範囲

2026-10-11時点：既存I2C単体3件、追加ホストテスト7件、既存リセット4件、既存AUTOモード5件が通過。
NNの基本制御3件も通過しました。NN学習・精度比較2件はPyTorch未導入のため実行できていません。
実CPU上のPSC-OSサービスによる既知入力への返信と、MuJoCoの20周期（100ms）の統合テストが通過しました。実行時間はブート込み約236秒です。
GUIからの起動・実際のI2C制御開始・停止・再起動・起動途中の停止も通過しました。

GUI検証の再実行（デスクトップが必要）：

```bash
myenv/bin/python balance-robot-mujoco-sim/tests/gui_psc_smoke.py \
  /path/to/cocotb-env/bin/python
```

実機外部ピン、通常PSC-OSシェル、NPU推論、長時間倒立安定性、実時間性能は未検証です。

PSC-OSは `make sync-psc-os` で `~/Program/PSC_RV32I/PSC-ONE/software/os/` から更新済みです。
シェルのI2C受信表示、ロボット連携サービス、起動表示の更新を含みます。
更新後の `external/PSC-ONE` は `SIM_USER=minimal` で通常カーネルと
`PSC_ROBOT_SIM` 専用イメージのビルドを確認済みです（`build/psc-os-sync/`）。
更新後の構成でもinteractiveモードの20周期（100ms）のRTL統合テストが通過しました。
strictモードでも3周期（15ms）のRTL統合テストが通過しました。
interactive制御ループの実測は4.005秒（OSブートを除く）、ブート込みのRTL実行は約69.4秒でした。
実時間動作には達していません。PSC関連ホストテスト18件、既存AUTO 5件、Reset 4件、
I2C単体RTLテスト3件も通過しています。操作検証は描画なしのQtウィジェットテストで、
Speed/Yaw、AUTO進行、通信待ち中の操作と古い応答の破棄を確認しています。
実測ログは `build/psc-interactive-check/interactive-20.log` にあります。
