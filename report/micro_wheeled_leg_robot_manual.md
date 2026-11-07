# Micro Wheeled leg Robot 倒立制御マニュアル

`Micro-Wheeled_leg-Robot-mujoco-sim/` のMuJoCoシミュレーターで、倒立制御を起動・操作する手順です。`balance` モードでは胴体を固定せず、左右の車輪トルクでバランスを取り、脚の目標角度を−0.10 rad（約−5.7度）に保持して、少し伸ばした姿勢にします。

## 倒立制御の起動

ターミナルで次を実行します。

```bash
cd /home/haruhiko/Program/PSC_Robot
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --mode balance
```

ウィンドウが開くと制御を開始します。画面左上に `BALANCE`、`ACTIVE`、現在の `Pitch` が表示されます。実行時間を指定しない場合、ウィンドウを閉じるまで動作します。

現在のモデルは約11.55度の前傾姿勢で釣り合う設定です。重心と荷重による脚サーボのたわみに合わせたモデル専用の姿勢基準であり、垂直から傾いていること自体は異常ではありません。倒立後の胴体原点の高さは約40.8 mmで、従来の姿勢より約14.3 mm高くなっています。

## 画面操作

| 操作・表示 | 内容 |
| --- | --- |
| `R` キー | ロボットの姿勢と制御状態を初期化し、倒立制御を再開 |
| マウス | 視点の回転・ズーム |
| ウィンドウを閉じる | シミュレーションを終了 |
| `ACTIVE` | 倒立制御が有効 |
| `FALL STOP - press R to reset` | 転倒判定により車輪出力を停止。`R`で復帰 |

胴体の鉛直方向からの傾斜が45度を超えると転倒と判定し、車輪出力を停止します。停止中も脚の位置保持は継続します。前進・旋回の操作、脚高さ変更、ジャンプは未実装です。

## 画面なしで実行して結果を保存

以下のコマンドはすべてリポジトリのルートで実行します。30秒分のシミュレーションを行い、計測結果をJSONに保存する例です。`--seconds` は実時間ではなくシミュレーション時間です。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py \
  --mode balance --headless --seconds 30 \
  --report report/balance_run.json
```

最終状態の画像も保存する場合は、次を実行します。画像出力にはEGLが利用できる環境が必要です。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py \
  --mode balance --headless --seconds 30 \
  --report report/balance_run.json \
  --render report/balance_run.png
```

同じ保存先を指定して再実行するとファイルを上書きします。JSONは終了時にターミナルにも出力されます。

| JSONの項目 | 読み方 |
| --- | --- |
| `simulated_seconds` | 実行したシミュレーション時間 |
| `warnings` | MuJoCoの警告数 |
| `final_pitch_deg` | 終了時のピッチ角度 |
| `chassis_displacement_m` | 初期位置からの胴体変位。X・Y・Z順、単位m |
| `balance.hip_target_rad` | 脚サーボの目標角度。現在は−0.10 rad |
| `balance.final_chassis_height_m` | 終了時の胴体原点の床からの高さ、単位m |
| `balance.fallen` | 転倒判定が発生すると `true` |
| `balance.max_abs_wheel_torque_nm` | 車輪トルクの絶対値の最大値、単位N m |
| `balance.final_mean_wheel_speed_rad_s` | 終了時の左右平均車輪角速度、単位rad/s |

## 起動オプション

| オプション | 内容 |
| --- | --- |
| `--mode balance` | 倒立制御を実行 |
| `--mode supported` | 胴体を固定し、脚と車輪を動かす機構確認デモ。省略時の既定値 |
| `--mode free` | 胴体を自由にし、脚の角度だけを保持。倒立制御なし |
| `--headless` | ウィンドウを開かずに実行 |
| `--seconds 30` | 実行するシミュレーション時間。正の有限値を指定 |
| `--report パス` | 終了時の計測結果をJSONで保存 |
| `--render パス` | 最終状態の画像をPNGで保存 |

`--seconds` を省略すると、画面なしでは10秒、GUIでは時間制限なしになります。倒立させる場合は必ず `--mode balance` を指定してください。

## 依存関係と動作確認

既存の `myenv/` を利用します。必要なPythonパッケージが不足している場合は次で導入します。通常の起動ではCAD変換用ライブラリは不要です。

```bash
myenv/bin/python -m pip install -r Micro-Wheeled_leg-Robot-mujoco-sim/requirements.txt
```

GUIにはデスクトップとOpenGLが必要です。画面を利用できない環境では、まず `--headless` を付け、`--render` を指定せずに実行してください。

倒立制御の回帰テストは次で実行できます。

```bash
myenv/bin/python -B Micro-Wheeled_leg-Robot-mujoco-sim/test_balance.py -v
```

テスト対象は30秒の倒立、制御なしとの比較、初期傾斜と外力からの回復、転倒停止とリセットなどです。

## 制御の設定と対象範囲

制御は1 msごとにピッチ、ジャイロY軸、平均車輪角度、平均車輪角速度から各車輪のトルクを計算します。現在の車輪トルク上限は各±0.03 N mです。

脚目標 `hip_target_rad`、姿勢基準 `pitch_reference_deg`、フィードバックゲインは [balance_controller.py](../Micro-Wheeled_leg-Robot-mujoco-sim/balance_controller.py) の `BalanceGains`、物理モデルとアクチュエータ設定は [scene.xml](../Micro-Wheeled_leg-Robot-mujoco-sim/scene.xml) にあります。ゲインはこのモデル向けに調整した値であり、LQRの最適化計算で求めた値ではありません。モデルの質量・重心・脚設定を変更した場合は再調整が必要です。

この制御はMuJoCo内のモデル専用です。質量・慣性・摩擦・駆動特性は仮設定で、PSC-ONEや実機には接続していません。

詳細は [シミュレーターREADME](../Micro-Wheeled_leg-Robot-mujoco-sim/README.md) と [モデル変換・実行結果の報告書](micro_wheeled_leg_robot_mujoco_report.md) を参照してください。
