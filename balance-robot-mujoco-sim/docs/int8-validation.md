# INT8二輪倒立制御の検証（2026-10-09）

既存の学習済み **4→16→16→1 / LeakyReLU(0.1)** を再学習せずFull Integer Quantizationし、MuJoCoで倒立できた。静止・外乱・緩やかな走行の18条件で、既存FP32と同じく各20秒完走、転倒0回。検証した構造の中で最小のINT8モデルを [`models/ROBOT.TFL`](../models/ROBOT.TFL) に保存した。姿勢のRMSはFP32より悪化しており、全ての制御品質が同等という結果ではない。

PSC-ONEの現行ランタイムはLeakyReLUを受け付けない。そのため、元モデルと数学的に等価な **4→32→32→1 / ReLU** 版も [`models/ROBOTFC.TFL`](../models/ROBOTFC.TFL) に保存した。この版はPSC-ONEの未変更CPUランタイムをホストで実行し、prepare成功を確認した。独立評価からの入力を含む5,123入力では、LiteRTとの出力差が14入力で1 LSBあり、完全一致ではない。今回、PSC-ONEのコード変更、RTL実行、実機測定は行っていない。

## 評価条件と結果

- MuJoCo 3.14.0、NumPy 2.5.3、LiteRT 2.3.0、TensorFlow変換器2.20.0。
- 制御200 Hz、物理刻み25 µs。既存の入力フィルター、±500 rad/s制限、Yaw合成を共用。
- 評価seed=20261009。静止、外乱、ランプ走行、急な速度切替を各6条件、各20秒上限。初期pitchは符号を交互にして0.04〜0.16 rad、heading・pitch rate・rollも変更。同じ初期値・指令・外乱で各制御器を独立走行。
- 外乱は7.0〜7.1秒に±0.015 / 0.025 / 0.035 N·m。初期姿勢のpitch軸に沿う固定世界座標軸へ、機体トルクを加える。
- 復帰時間は外乱終了後、機体傾斜2度以内を0.5秒継続した区間の開始まで。転倒は左右方向を含む機体傾斜45度超え。角度RMS・最大角は各試行の初期過渡も含む。

通常条件18試行（静止6、外乱6、±0.3 m/sまでの緩やかな走行6）：

| 指標 | LQR | 既存NN FP32 | NN INT8 / ROBOT | 等価ReLU INT8 / ROBOTFC |
|---|---:|---:|---:|---:|
| 平均倒立維持時間 [s] | 20.000 | 20.000 | 20.000 | 20.000 |
| 転倒 / 18 | 0 | 0 | 0 | 0 |
| 転倒率 | 0% | 0% | 0% | 0% |
| pitch RMSの試行平均 [deg] | 1.594 | 1.503 | 2.067 | 2.028 |
| 最大絶対pitch [deg] | 24.324 | 15.741 | 19.427 | 17.921 |
| LQRに対する制御出力RMSEの試行平均 [rad/s] | 0 | 0.075 | 1.220 | 1.164 |
| 外乱後の復帰成功 / 6 | 6 | 6 | 6 | 6 |
| 平均復帰時間 [s] | 0.5125 | 0.5108 | 0.4650 | 0.6533 |

制御出力誤差は、それぞれの走行中に実際に観測された**同じ状態・同じフィルター値**に対するLQR出力と比較したもの。別々の軌跡の時系列を直接引き算していない。INT8の復帰時間が一部短いことだけから、外乱耐性の改善とは判断しない。

急な速度切替（0→±0.6→∓0.6→0 m/s）の6試行は、LQR・FP32・両INT8とも6/6転倒。転倒までの平均はLQR 5.1325秒、FP32 7.6508秒、ROBOT 7.6483秒、ROBOTFC 7.6475秒。全24試行で見ると各方式の転倒率は25%。既存制御でも失敗する条件を除外して成功率を表示しているわけではなく、上表と分けて記録した。

エピソード別の値、初期値、外乱、転倒条件、依存バージョン、シーンとモデルのSHA-256は [`int8-native-evaluation.json`](int8-native-evaluation.json) と [`int8-relu-evaluation.json`](int8-relu-evaluation.json) に保存。共有状態による推論誤差の再検証用に、それぞれの `.states.npz` も保存した。

## 推論誤差と推論時間

FP32は「既存のFP32学習済み重みを使うNN」を指す。既存 `NNPredictor` は重みをロード時にfloat64へ変換してNumPy演算するため、比較時もその動作を変更していない。TensorFlowのfloat32グラフ／TFLite FP32への変換誤差は別途確認した。

| 指標 | 元のLeakyReLU版 | 等価ReLU版 |
|---|---:|---:|
| 量子化前TFLite FP32対元NNの最大誤差 [rad/s] | 0.00000327 | 0.00000422 |
| INT8対元NNのRMSE [rad/s] | 1.3459 | 1.0050 |
| INT8対元NNの平均絶対誤差 [rad/s] | 1.2039 | 0.8612 |
| INT8対元NNの最大絶対誤差 [rad/s] | 3.6998 | 3.2552 |
| INT8対元NNの絶対誤差99 percentile [rad/s] | 2.7423 | 2.4210 |
| 評価入力の量子化範囲外率（各4特徴） | 全て0% | 全て0% |
| 出力がINT8端点に達した割合 | 0.0500% | 0.0583% |

各INT8評価でLQR・FP32・該当INT8の軌跡から均等抽出した12,000状態について、FP32とINT8を同一入力で比較。急な速度切替の転倒までの状態も含む。二つのINT8版で軌跡が異なるため、両列の抽出状態は完全に同一ではない。量子化前の確認は代表データの最大1,024状態で行った。

| 呼び出し | 平均 [µs] | p95 [µs] |
|---|---:|---:|
| LQR式 | 3.021 | 3.134 |
| 元NN | 6.723 | 6.799 |
| ROBOT INT8 | 8.108 | 8.314 |
| ROBOTFC INT8 | 8.236 | 8.427 |

ホスト上の4,000回測定。100回ウォームアップ、1スレッド、既定delegateを無効化したLiteRT CPU整数カーネルを使用。NNには正規化・量子化・set/invoke/get・出力変換を含む。LQRは同じ4特徴から出力を求める式。センサー取得、フィルター更新、MuJoCoステップ、GUI描画を含まない。PSC-ONE上の実行時間ではない。この小さいNNではPython呼び出しの負担もあり、ホストでINT8がFP32より速いという結果ではない。

## 量子化・ファイル・演算量

[公式のFull Integer Quantization手順](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)に従い、`TFLITE_BUILTINS_INT8`、INT8入力、INT8出力を指定。全結合の重みと中間活性化はINT8、biasと積和の蓄積はINT32。FLOAT32テンソルやFLOAT32推論へのフォールバックはない。出力だけを丸める疑似量子化ではない。

代表データはseed=8128、16条件×LQR/既存NNの32走行から収集した53,224個の実MuJoCo状態。評価seedとは分離し、合成した状態で水増ししていない。seed=42で4,096個を抽出し、各特徴の実測最小／最大を含む合計4,102状態で較正。分布・収集条件を [`int8_calibration.json`](../models/int8_calibration.json)、状態を `.npz`、変換条件と全テンソル量子化情報を各モデルの `.json` に保存。

| モデル | 構造・活性化 | dense MAC/推論 | パラメーター数 | TFLite FP32 [B] | TFLite INT8 [B] |
|---|---|---:|---:|---:|---:|
| ROBOT | 4→16→16→1、LeakyReLU | 336 | 369 | 3,304 | 2,768 |
| ROBOTFC | 4→32→32→1、ReLU（等価展開） | 1,184 | 1,249 | 6,588 | 3,424 |
| 追加学習候補32幅 | 4→32→32→1 | 1,184 | 1,249 | 未生成 | 未生成 |
| 追加学習候補64幅 | 4→64→64→1 | 4,416 | 4,545 | 未生成 | 未生成 |

MACは全結合の積和だけを数え、bias、活性化、正規化、再量子化を含まない。ファイルサイズはFlatBuffer全体で、JSON側の正規化情報は別。元の `robot_nn.npz` は4,035 B、`robot_nn.pt` は5,253 Bで変更なし。

演算子は次のとおり。

- **ROBOT:** FULLY_CONNECTED v4 ×3（融合活性化NONE）、LEAKY_RELU v1 ×2。
- **ROBOTFC:** FULLY_CONNECTED v4 ×3（融合活性化RELU、RELU、NONE）。独立したRELU演算子は不要。

元の16幅で倒立維持目標を満たしたため、32幅・64幅の再学習やQATは行っていない。ROBOTFCは量子化失敗を幅の拡大で補う試みではなく、PSC-ONEの対応演算子へ変換するための等価表現である：

```text
LeakyReLU(z) = ReLU(z) - 0.1 * ReLU(-z)
```

各隠れ層に正負両方のReLUを保持し、次の全結合の重みに上式の係数を織り込む。元の重みから代数的に生成するため再学習していない。量子化後まで両グラフが同じ出力になるとは仮定せず、別々に制御評価した。

## 入出力契約とPSC-ONE移植

特徴量順序は `pitch, filtered_pitch_rate, filtered_wheel_angular_velocity, target_linear_velocity`。元のNNと同じ平均・標準偏差を `.json` に固定し、モデルSHA-256と結びつける。GUIでFP32を再学習しても、既存INT8モデルの正規化値は勝手に入れ替えない。INT8を更新するには明示的な再変換が必要。

```text
normalized = (state - mean) / scale
q_in = clip(round_to_even(normalized / input_scale) + input_zero_point, -128, 127)
q_out = TFLite_INT8.invoke(q_in)
motor_velocity = (q_out - output_zero_point) * output_scale_TFLite
                 * output_scale_original + output_mean_original
```

両モデルの入力は固定 `[1,4]`、出力は固定 `[1,1]`。今回の入力scaleは0.16758327186107635、zero pointは1。出力scaleは0.04138699546456337、zero pointは-65。元の出力scaleは10.287489891052246、meanは2.3699944019317627。出力1 LSBは約0.425768 rad/s。符号、zero point、飽和を適用してからINT8へcastし、逆変換も必ず行う。

センサーの物理量からの正規化とモーター物理量への逆変換は、既存NN同様モデル外のI/O処理。NNの入力・出力テンソルはINT8だが、MuJoCo側のこのI/O処理は浮動小数点。将来のPSC-ONE制御アプリにも、同じフィルターとこの契約に沿うI/O処理が必要になる。

PSC-ONEの `software/os/src/api/tflite` の**現在のソースをそのまま**ホスト共有ライブラリとしてビルドした。成果物は `/tmp` に出力し、PSC-ONEへ書き込んでいない。

| 移植確認 | ROBOT | ROBOTFC |
|---|---|---|
| 既存ランタイムprepare | -202 / unsupported（LeakyReLU） | 0 / 成功 |
| LiteRTとの出力バイト一致（独立評価入力を含む） | 対象外 | 5,109 / 5,123 |
| 最大差 [LSB] | 対象外 | 1 |
| tensor arena使用量 | 対象外 | 332 B / 容量4,096 B |

5,123入力は、独立評価で取得した4,096状態、1,024個のランダムINT8ベクトル、全-128／全0／全127の3境界ベクトル。較正データから4,096状態を選んだ最初の試験は5,123/5,123一致だったが、独立評価のデータでは完全一致が成立しなかった。1 LSBは約0.425768 rad/sに相当する。LiteRTのBUILTIN_REFでもPSCとの不一致は165入力あり、参照カーネルに切り替えるだけでは解消しなかった。

入力コード `[4,1,1,0]` を層ごとに確認すると、第1・第2層は一致し、最終層の再量子化だけが異なった。積和値は-156、有効倍率は0.003194300846765005、実数積は-0.4983109320953408。PSCの既存double rounding実装は整数multiplier=1756085462、shift=-8を使い、最初の丸めで-128、次の右シフト丸めで-1となり、zero point=-65を足して-66を出す。LiteRTは-65を出す。この例では二段階丸めによる差と確認できた。全不一致ベクトルについての層別証明は行っていない。

arena値はランタイム全体のRAM使用量ではない。ソースSHA-256、差が出た入力例と結果は [`int8-psc-native.json`](int8-psc-native.json)、[`int8-psc-relu.json`](int8-psc-relu.json) を参照。**出力バイト完全一致を要求するホストテストはFAIL**であり、モデルの読み込み・実行に成功したこととは区別する。

この差の制御への影響を確認するため、変更していないPSCのCPUランタイムをホストライブラリとしてMuJoCoへ接続し、同じ24条件を追加実行した。通常18条件は各20秒完走・転倒0回、pitch RMS平均1.9804度、外乱6条件は全て復帰。急な速度切替6条件は全て転倒、平均7.6467秒。PSC-ONE側のコードは変更していない。この追加確認はホスト上での制御評価であり、RV32の命令実行時間や実機の確認ではない。

**推奨:** MuJoCoで最小のまま使うなら4→16→16→1のROBOT。PSC-ONEの既存ランタイムを変更せずに読み込み・実行する移植候補は4→32→32→1の等価ReLU版ROBOTFC。ただしLiteRTとの厳密な出力互換性には上記の差が残る。ファイル名はいずれもFAT32の8.3形式に収まる。実SoCでのレイテンシ、実ロボットの安定性、長時間運転や未評価の外乱については今回の結果から保証しない。

## GUIと再実行

`balance-robot-mujoco-sim` から実行する。通常のGUI環境にはLiteRTだけを追加し、TensorFlowは変換用の別Python 3.13環境に導入した。

```bash
python -m pip install -r requirements-int8.txt
python src/simulation/simulate_robot.py
```

Controllerの **NN INT8** を選択。起動時に `models/ROBOT.TFL` と `models/ROBOT.json` を読み込む。既存の **Neural Network** がFP32モード。LQRとReinforcement Learningの選択・学習機能も維持。INT8モデル／ランタイムがない、またはモデルのハッシュが不一致ならINT8だけ無効にする。

較正・変換・評価の再現（`TF_PYTHON` はTensorFlow 2.20.0を導入したPythonのパス）：

```bash
python src/simulation/int8_experiments.py collect --duration 10 --repeats 4 \
  --seed 8128 --output models/int8_calibration.npz
"$TF_PYTHON" -m pip install -r requirements-int8-convert.txt
"$TF_PYTHON" src/simulation/convert_int8.py --calibration models/int8_calibration.npz \
  --output models/ROBOT.TFL
"$TF_PYTHON" src/simulation/convert_int8.py --calibration models/int8_calibration.npz \
  --relu-only --output models/ROBOTFC.TFL
python src/simulation/int8_experiments.py evaluate --int8 models/ROBOT.TFL \
  --output docs/int8-native-evaluation.json
python src/simulation/int8_experiments.py evaluate --int8 models/ROBOTFC.TFL \
  --output docs/int8-relu-evaluation.json
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p 'test_*.py'
python tests/gui_int8_smoke.py
```

PSC-ONE側を変更しないホスト互換確認：

```bash
python tests/check_psc_int8.py --os /path/to/PSC-ONE/software/os \
  --build /tmp/psc-robot-port-native --model models/ROBOT.TFL \
  --states docs/int8-native-evaluation.states.npz --expect-unsupported
python tests/check_psc_int8.py --os /path/to/PSC-ONE/software/os \
  --build /tmp/psc-robot-port-relu --model models/ROBOTFC.TFL \
  --states docs/int8-native-evaluation.states.npz --rollouts
```

後者はJSONと追加走行結果を保存した後、厳密なバイト一致のassertで非ゼロ終了する。既知の1 LSB差を無視してPASSにするようなテストの緩和は行っていない。

39件の自動テストPASS。実画面でNN INT8→FP32→LQR→RL→NN INT8と切り替え、AUTO、Reset、実INT8推論によるモーター指令、OpenGL描画、正常終了を確認した。元のFP32モデル・重み・学習データは保存したまま。開始時点から変更済みだった `robot_rl.rl.zip` も、開始時のSHA-256を維持している。追加学習、git add/commit/push、既存ファイルの削除は行っていない。
