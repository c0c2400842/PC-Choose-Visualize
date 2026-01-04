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
