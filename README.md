# PC Cost-Performance Analyzer (PCA & Pareto Optimization)

データサイエンスの手法（主成分分析とパレート最適化）を用いて、自分に最適なPCを見つけるためのデスクトップアプリです。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 主な機能

- **主成分分析 (PCA)**: 複雑なスペックデータを「総合性能」と「特化方向」の2軸に集約。
- **パレート最適化**: 性能と価格のバランスが取れた「効率的な選択肢」を自動抽出。
- **インタラクティブ分析**: スライダーを動かすだけで、自分の好みに合わせたランキングをリアルタイム更新。
- **CSV管理**: 独自のPCデータを読み込み・編集・保存可能。

## 🚀 セットアップ

### 1. リポジトリをクローン
```bash
git clone https://github.com/YOUR_USERNAME/pc-cost-performance-analyzer.git
cd pc-cost-performance-analyzer
```

### 2. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

### 3. アプリの起動
```bash
python pc_visualize_app.py
```

## 📊 分析の仕組み

1. **データの標準化**: 単位の異なるスペックを公平に比較。
2. **PCA実行**: データの寄与度を解析し、PC2（第2主成分）が「CPU重視」か「SSD重視」かなどを自動判定。
3. **理想点からの距離**: 「最高性能かつ最低価格」の架空の点（理想点）からの距離を計算し、スコア化。

## 🛠 使用技術

- **Language**: Python 3.13
- **GUI**: PySide6
- **Analysis**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib

## 📝 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
