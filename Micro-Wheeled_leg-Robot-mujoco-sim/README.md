# Micro Wheeled leg Robot MuJoCo simulation

`external/Micro-Wheeled_leg-Robot/1.RobotModel/OriginalRobotModel.stp` を変換したMuJoCoモデルです。73個のCAD部品を9個の剛体にまとめ、左右の4節リンク、脚サーボ2軸、車輪モーター2軸を実装しています。

## 起動

リポジトリのルート `/home/haruhiko/Program/PSC_Robot` で実行します。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py
```

標準の `supported` モードは胴体を固定し、左右の脚と車輪を周期的に動かすデモです。空中での機構確認用であり、自立動作ではありません。マウスで視点を変更し、`R` で初期状態に戻します。ウィンドウを閉じると終了します。

床接触と重力を含む自由運動は次で実行します。脚角度を保持するだけで、倒立制御はありません。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --mode free
```

倒立制御の最小版は `balance` モードで実行します。胴体は固定せず、脚の目標角度を−0.10 rad（約−5.7度）に保持して少し伸ばし、左右の車輪トルクでバランスを取ります。`R`で制御状態もリセットします。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --mode balance
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --mode balance --headless --seconds 30 --report Micro-Wheeled_leg-Robot-mujoco-sim/results/balance.json
```

### 最小倒立制御

元リポジトリの `3.Software/wl_pro_robot/wl_pro_robot.ino` にある `lqr_balance_loop()` の4状態フィードバックをPythonで実装しています。左右それぞれの車輪に同じトルクを与えます。

```text
tau = 0.15 * (pitch - pitch_reference)
    + 0.01 * gyro_y
    + 0.0005 * (mean_wheel_angle - initial_mean_wheel_angle)
    + 0.002 * mean_wheel_speed
```

角度はrad、角速度はrad/s、出力はN mです。ゲインはこのモデルで調整した値で、LQRの最適化計算から求めた値ではありません。元コードの電圧指令や度単位のゲインをそのまま使用していません。電気系・FOCの再現も含みません。

姿勢にはMuJoCoの姿勢センサー、角速度にはジャイロ、車輪角度・速度には関節状態を使い、物理ステップごと（1 ms）に制御します。姿勢推定誤差・センサーノイズは加えていません。左右の車輪軸がともに+Yなので、元コードのモーター配線由来の符号は置き換えています。

姿勢基準は **11.55度の前傾**です。現モデルの重心と荷重による脚サーボのたわみを含む平衡姿勢に合わせています。脚の目標角度は−0.10 radですが、実際の関節角度は荷重でずれます。実機の基準値や測定値ではありません。モデルの質量・慣性・脚設定を変えた場合は再調整が必要です。

車輪出力は各±0.03 N mに制限します。胴体の傾斜が45度を超えると車輪出力を停止し、`R`を押すまで復帰しません。脚位置の保持は継続します。前進指令、旋回、脚高さ変更、ジャンプ、積分補償、重心の自動調整は未実装です。

検証は次で再実行できます。

```bash
myenv/bin/python -B Micro-Wheeled_leg-Robot-mujoco-sim/test_balance.py -v
```

30秒の倒立、制御なしとの比較、初期傾斜±5度と前後方向の外力（0.15 Nを0.1秒）からの回復、転倒停止とリセットを検証します。倒立中に胴体・脚が床に接触していないことも確認します。30秒の標準実行では最終ピッチ11.548度、前後変位約+10.9 mm、最終平均車輪速度0.000110 rad/s、MuJoCo警告0件でした。胴体原点の高さは約40.8 mmで、従来の脚目標0 radの姿勢（約26.5 mm）から約14.3 mm上がっています。

画面なしで10秒の動作を確認し、画像と測定結果を保存できます。

```bash
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --headless --seconds 10 --render Micro-Wheeled_leg-Robot-mujoco-sim/results/supported.png --report Micro-Wheeled_leg-Robot-mujoco-sim/results/supported.json
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/simulate.py --mode free --headless --seconds 10 --render Micro-Wheeled_leg-Robot-mujoco-sim/results/free.png --report Micro-Wheeled_leg-Robot-mujoco-sim/results/free.json
```

## ファイル

| ファイル | 内容 |
| --- | --- |
| `scene.xml` | 自由浮遊胴体、床、関節、アクチュエータ、IMUを含むMJCF |
| `scene_supported.xml` | 胴体固定の機構確認用MJCF |
| `meshes/` | STEP由来の73個のバイナリSTL、単位m |
| `simulate.py` | GUIとヘッドレス実行、画像・計測結果出力 |
| `balance_controller.py` | 元コードの4状態フィードバックを用いた最小倒立制御 |
| `test_balance.py` | 倒立・外乱回復・転倒停止・リセットの物理回帰テスト |
| `convert_step.py` | 部品分割、メッシュ生成、MJCFの再生成 |
| `conversion_manifest.json` | 元ファイルのSHA-256、部品名と剛体の対応、座標変換 |
| `results/` | 実行結果とレンダリング画像 |

## 依存関係と再変換

動作確認環境はPython 3.14、MuJoCo 3.15.0です。メッシュは同梱しているので、通常の実行にCADライブラリは不要です。

```bash
myenv/bin/python -m pip install -r Micro-Wheeled_leg-Robot-mujoco-sim/requirements.txt
```

STEPを再変換する場合はOpen CASCADEのPythonバインディングを追加します。

```bash
myenv/bin/python -m pip install -r Micro-Wheeled_leg-Robot-mujoco-sim/requirements-convert.txt
myenv/bin/python Micro-Wheeled_leg-Robot-mujoco-sim/convert_step.py
```

変換は付属STEPの部品構成専用です。部品順と代表的な部品名を検査しますが、任意のSTEPに対する自動関節推定ではありません。元CADを更新した場合は部品対応と支点座標を再確認してください。

## モデルの精度

外形・配置・車輪半径18.5 mm・関節軸はCAD由来です。CADのmmをmに変換し、前方をX、左をY、上をZとしています。左右各4個のヒンジと、4節リンクを閉じる`connect`制約を使っています。4個のアクチュエータの順序は左脚位置、右脚位置、左車輪トルク、右車輪トルクです。脚位置の単位はrad、車輪トルクはN mです。関節角度0はCADの組立姿勢です。

合計質量0.320 kg、各剛体の質量配分、慣性、摩擦、減衰、モータートルク上限、関節可動域は仮設定です。慣性は剛体ごとの外接箱から計算しています。外観の色も表示用設定です。内蔵ばね、配線、ギアバックラッシュは再現していません。

衝突形状は車輪の円柱と胴体・脚の簡易形状を使い、床との接触のみを計算します。自己衝突は無効です。詳細メッシュの凹形状をそのまま衝突判定に使ったモデルではありません。形状を含まない基板部品1件はスキップしています。

既存の`balance-robot-mujoco-sim/`やPSC-ONEの制御器には接続していません。`balance`モードはこのモデル専用の最小倒立制御です。実機に近い制御評価には質量・重心・慣性・モーター特性の同定が必要です。

CAD出典: [MuShibo/Micro-Wheeled_leg-Robot](https://github.com/MuShibo/Micro-Wheeled_leg-Robot)。生成メッシュはこのCADの派生物です。結果の詳細は[`report/`の報告書](../report/micro_wheeled_leg_robot_mujoco_report.md)を参照してください。
