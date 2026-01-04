# PCコスパ分析 統合アプリ

データサイエンスの手法（主成分分析）を用いて、PCのスペック構成を可視化し、ユーザーの用途に合わせた最適なPC選択を支援するデスクトップアプリケーションです。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)


## 🚀 セットアップ

### 1. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

必要なパッケージ:
- PySide6
- numpy
- pandas
- scikit-learn
- matplotlib

### 2. アプリの起動
```bash
python pc_visualize_app.py
```

## 📊 分析の仕組み

### 1. **標準化PCA（Principal Component Analysis）**
- **標準化**: 各特徴量（CPU、GPU、RAM、SSD）のスケールを揃えるため、平均0・分散1に正規化
- **主成分抽出**: 
  - **PC1**: 通常は全特徴量の総合力を表す軸（総合性能）
  - **PC2**: 特徴量間の対立軸を表す軸（例：GPU重視 ↔ CPU重視）
- **動的ラベリング**: 固有ベクトルの寄与度から、各主成分が何を意味するかを自動判定

### 2. **嗜好ベクトルによるスコアリング**
- ユーザーが設定した重み（構成バランス）をPCA空間上のベクトルとして扱う
- 各PCの座標と嗜好ベクトルの類似度を計算し、適合スコアを算出
- スコア計算式: `0.5 × (PC1の正規化値) + 0.5 × (構成バランス重み × PC2の正規化値)`

### 3. **予算フィルタリング**
- 予算内のPCのみを対象にスコア計算を実施
- 予算内にPCがない場合は、全PCから最適なものを提示（予算外として明示）

## 🎨 画面構成

### 左パネル（PCA情報）
- PC1・PC2の寄与率と累積寄与率をプログレスバーで視覚化
- 各スペック（CPU、GPU、RAM、SSD）の寄与度テーブル

### 中央パネル（グラフ）
- PCA空間上にPCをプロット
- 推奨PCを⭐マークで強調表示
- 予算外のPCはグレーアウト

### 右パネル（推奨PC）
- 推奨PCのモデル名、価格、スペック、適合スコアを大きく表示
- 現在選択中のプリセットを表示

## 💡 使い方

1. **CSVタブ**でPCデータを管理（追加・編集・保存）
2. **コスパ分析タブ**で「このデータで分析」ボタンをクリック
3. プリセットボタンまたはスライダーで嗜好を調整
4. 推奨PCが右パネルに表示され、グラフ上でも⭐マークで強調
5. グラフ上の点をクリックすると詳細情報を確認可能

## 🛠 使用技術

- **Language**: Python 3.10+
- **GUI Framework**: PySide6 (Qt for Python)
- **Data Analysis**: scikit-learn (PCA), pandas, numpy
- **Visualization**: matplotlib
- **Data Management**: CSV

## 📂 ファイル構成

```
.
├── pc_visualize_app.py      # メインアプリケーション
├── pc_data.csv              # サンプルPCデータ
├── requirements.txt         # 依存ライブラリ一覧
├── README.md                # このファイル
└── last_csv_path.txt        # 前回使用したCSVパス（自動生成）
```



## 📝 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 🤝 貢献

バグ報告や機能提案は Issue でお願いします。プルリクエストも歓迎します！
