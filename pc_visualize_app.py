import sys
import csv
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit,
    QFileDialog, QMessageBox, QSlider,
    QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QFrame, QProgressBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===== matplotlib optional =====
HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    # 日本語フォントの設定 (Windows: MS Gothic, Mac: AppleGothic, etc.)
    plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

LAST_CSV_FILE = "last_csv_path.txt"

# プリセット定義（w_pc2: 構成バランス CPU重視↔GPU/容量重視）
# 性能レベルは予算スライダーで制御するため、w_pc1は廃止
PRESETS = {
    "プログラマー": {
        "w_pc2": 80,
        "color": "#1976D2",  # 青
        "description": "CPU・RAM重視（コンパイル・開発向け）"
    },
    "ゲーマー": {
        "w_pc2": -90,
        "color": "#D32F2F",  # 赤
        "description": "GPU・ストレージ重視（ゲーム向け）"
    },
    "動画編集者": {
        "w_pc2": -60,
        "color": "#7B1FA2",  # 紫
        "description": "GPU・容量重視（動画編集向け）"
    },
    "一般ユーザー": {
        "w_pc2": 0,
        "color": "#388E3C",  # 緑
        "description": "バランス型（一般用途）"
    },
    "AI・データ分析": {
        "w_pc2": 50,
        "color": "#FFA000",  # オレンジ
        "description": "CPU重視（計算処理向け）"
    }
}

# ================================
# フォントサイズ設定（一箇所で管理）
# ================================
FONT_SCALE = 1.0  # 読みやすさ重視で1.0に設定

class FontSize:
    """フォントサイズを一括管理するクラス"""
    # 左パネル（PCA情報）
    PCA_TITLE = int(18 * FONT_SCALE)
    PCA_LABEL = int(12 * FONT_SCALE)
    PCA_VALUE = int(22 * FONT_SCALE)
    PCA_CUMSUM = int(14 * FONT_SCALE)
    PCA_CONTRIB_TITLE = int(14 * FONT_SCALE)
    PCA_TABLE = 9
    
    # 右パネル（推奨PC）
    REC_TITLE = int(20 * FONT_SCALE)
    REC_PC_NAME = int(18 * FONT_SCALE)
    REC_PRICE = int(32 * FONT_SCALE)
    REC_SPECS = int(13 * FONT_SCALE)
    REC_SECTION_TITLE = int(16 * FONT_SCALE)
    REC_SCORE = int(18 * FONT_SCALE)
    REC_PRESET_LABEL = int(13 * FONT_SCALE)
    REC_PRESET = int(16 * FONT_SCALE)
    REC_WEIGHT = int(12 * FONT_SCALE)
    REC_INFO = int(13 * FONT_SCALE)
    REC_SUBTITLE = int(11 * FONT_SCALE)
    
    # ボタン・コントロール
    BTN_MAIN = int(14 * FONT_SCALE)
    BTN_PRESET = int(13 * FONT_SCALE)
    PRESET_LABEL = int(14 * FONT_SCALE)
    SLIDER_LABEL = int(14 * FONT_SCALE)
    
    # グラフ
    GRAPH_AXIS = int(13 * FONT_SCALE)
    GRAPH_TITLE = int(14 * FONT_SCALE)
    GRAPH_LEGEND = int(11 * FONT_SCALE)

# ================================
# PCA情報パネル（左側固定）
# ================================

class PCAInfoPanel(QWidget):
    """PCA情報を常に表示する左側固定パネル"""
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(180)
        self.setStyleSheet("background-color: #F5F5F5; padding: 3px;")
        
        layout = QVBoxLayout(self)
        
        # ========== タイトル ==========
        title = QLabel("📊 主成分分析")
        title.setStyleSheet(f"""
            font-size: {FontSize.PCA_TITLE}px; 
            font-weight: bold; 
            color: #1976D2;
            margin-bottom: 3px;
        """)
        layout.addWidget(title)
        
        # ========== 説明テキスト ==========
        self.explanation = QLabel("データ読込後に\n軸の意味を自動判定")
        self.explanation.setStyleSheet(f"""
            font-size: {FontSize.PCA_LABEL}px; 
            color: #616161;
            background-color: #E3F2FD;
            padding: 5px;
            border-radius: 5px;
            border: 1px solid #90CAF9;
        """)
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        
        layout.addSpacing(5)
        
        # ========== PC1寄与率 ==========
        self.pc1_label = QLabel("PC1寄与率")
        self.pc1_label.setStyleSheet(f"font-size: {FontSize.PCA_LABEL}px; font-weight: bold;")
        layout.addWidget(self.pc1_label)
        
        self.pc1_value = QLabel("0.0%")
        self.pc1_value.setStyleSheet(f"font-size: {FontSize.PCA_VALUE}px; color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.pc1_value)
        
        self.pc1_bar = QProgressBar()
        self.pc1_bar.setRange(0, 100)
        self.pc1_bar.setValue(0)
        self.pc1_bar.setTextVisible(False)
        self.pc1_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #E0E0E0;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.pc1_bar)
        
        # ========== PC2寄与率 ==========
        layout.addSpacing(5)
        
        self.pc2_label = QLabel("PC2寄与率")
        self.pc2_label.setStyleSheet(f"font-size: {FontSize.PCA_LABEL}px; font-weight: bold;")
        layout.addWidget(self.pc2_label)
        
        self.pc2_value = QLabel("0.0%")
        self.pc2_value.setStyleSheet(f"font-size: {FontSize.PCA_VALUE}px; color: #2196F3; font-weight: bold;")
        layout.addWidget(self.pc2_value)
        
        self.pc2_bar = QProgressBar()
        self.pc2_bar.setRange(0, 100)
        self.pc2_bar.setValue(0)
        self.pc2_bar.setTextVisible(False)
        self.pc2_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                background-color: #E0E0E0;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        layout.addWidget(self.pc2_bar)
        
        # ========== 累積寄与率 ==========
        layout.addSpacing(3)
        
        self.cumsum_label = QLabel("累積寄与率: 0.0%")
        self.cumsum_label.setStyleSheet(f"""
            font-size: {FontSize.PCA_CUMSUM}px; 
            font-weight: bold; 
            color: #F57C00;
            background-color: #FFF3E0;
            padding: 4px;
            border-radius: 5px;
        """)
        layout.addWidget(self.cumsum_label)
        
        # ========== 区切り線 ==========
        layout.addSpacing(8)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #BDBDBD;")
        layout.addWidget(separator)
        layout.addSpacing(3)
        
        # ========== 各スペックの寄与表 ==========
        contrib_title = QLabel("📐 スペックの寄与度")
        contrib_title.setStyleSheet(f"font-size: {FontSize.PCA_CONTRIB_TITLE}px; font-weight: bold;")
        layout.addWidget(contrib_title)
        
        self.contrib_table = QTableWidget(4, 2)
        self.contrib_table.setHorizontalHeaderLabels(["PC1", "PC2"])
        self.contrib_table.setVerticalHeaderLabels(["CPU", "GPU", "RAM", "SSD"])
        self.contrib_table.horizontalHeader().setStretchLastSection(True)
        self.contrib_table.setMaximumHeight(180)
        self.contrib_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: white;
                gridline-color: #E0E0E0;
                font-size: {FontSize.PCA_TABLE}px;
            }}
            QHeaderView::section {{
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                padding: 3px;
                font-size: {FontSize.PCA_TABLE}px;
            }}
        """)
        # 列幅を均等に設定
        self.contrib_table.horizontalHeader().setSectionResizeMode(0, self.contrib_table.horizontalHeader().ResizeMode.Stretch)
        self.contrib_table.horizontalHeader().setSectionResizeMode(1, self.contrib_table.horizontalHeader().ResizeMode.Stretch)
        # 垂直ヘッダーの幅を小さく
        self.contrib_table.verticalHeader().setMaximumWidth(40)
        layout.addWidget(self.contrib_table)
        
        # 下部の余白
        layout.addStretch()
    
    def update_pca_info(self, pca, var_ratio, pc1_desc="", pc2_desc=""):
        """PCA結果を更新"""
        if len(var_ratio) == 0:
            return
        
        # 説明テキストの更新
        self.explanation.setText(f"PC1: {pc1_desc}\nPC2: {pc2_desc}")
        self.pc1_label.setText(f"PC1: {pc1_desc}")
        self.pc2_label.setText(f"PC2: {pc2_desc}")
        
        # PC1寄与率
        self.pc1_value.setText(f"{var_ratio[0]*100:.1f}%")
        self.pc1_bar.setValue(int(var_ratio[0]*100))
        
        # PC2寄与率
        if len(var_ratio) >= 2:
            self.pc2_value.setText(f"{var_ratio[1]*100:.1f}%")
            self.pc2_bar.setValue(int(var_ratio[1]*100))
            cumsum = np.cumsum(var_ratio)
            self.cumsum_label.setText(f"累積寄与率: {cumsum[1]*100:.1f}%")
        else:
            self.pc2_value.setText("0.0%")
            self.pc2_bar.setValue(0)
            self.cumsum_label.setText(f"累積寄与率: {var_ratio[0]*100:.1f}%")
        
        # 寄与度テーブルを更新
        components = pca.components_
        features = ['CPU', 'GPU', 'RAM', 'SSD']
        for i, feature in enumerate(features):
            # PC1
            pc1_val = components[0, i]
            pc1_item = QTableWidgetItem(f"{pc1_val:+.3f}")
            pc1_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if abs(pc1_val) > 0.4:
                pc1_item.setBackground(QColor("#C8E6C9") if pc1_val > 0 else QColor("#FFCDD2"))
            self.contrib_table.setItem(i, 0, pc1_item)
            
            # PC2
            if len(var_ratio) >= 2:
                pc2_val = components[1, i]
                pc2_item = QTableWidgetItem(f"{pc2_val:+.3f}")
                pc2_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if abs(pc2_val) > 0.4:
                    pc2_item.setBackground(QColor("#BBDEFB") if pc2_val > 0 else QColor("#FFE0B2"))
                self.contrib_table.setItem(i, 1, pc2_item)
            else:
                self.contrib_table.setItem(i, 1, QTableWidgetItem("0.000"))


# ================================
# 推奨PCパネル（右側固定）
# ================================

class RecommendationPanel(QWidget):
    """推奨PCを大きく表示する右側固定パネル"""
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(250)
        self.setStyleSheet("background-color: #FAFAFA; padding: 5px;")
        
        layout = QVBoxLayout(self)
        
        # ========== 総合評価1位PC ==========
        title = QLabel("🏆 あなたへの推奨PC")
        title.setStyleSheet(f"""
            font-size: {FontSize.REC_TITLE}px; 
            font-weight: bold; 
            color: #FF6F00;
            margin-bottom: 5px;
        """)
        layout.addWidget(title)
        
        subtitle = QLabel("（嗜好に最も近いPC）")
        subtitle.setStyleSheet(f"font-size: {FontSize.REC_SUBTITLE}px; color: #757575; margin-top: -5px; margin-bottom: 5px;")
        layout.addWidget(subtitle)
        
        # ========== PC名 ==========
        self.pc_name = QLabel("「このデータで分析」をクリック")
        self.pc_name.setStyleSheet(f"""
            font-size: {FontSize.REC_PC_NAME}px; 
            font-weight: bold; 
            color: #757575;
            background-color: #F1F8E9;
            padding: 10px;
            border-radius: 8px;
            border: 3px solid #FFD700;
        """)
        self.pc_name.setWordWrap(True)
        layout.addWidget(self.pc_name)
        
        # ========== 価格（超大きく） ==========
        self.pc_price = QLabel("―――")
        self.pc_price.setStyleSheet(f"""
            font-size: {FontSize.REC_PRICE}px; 
            font-weight: bold; 
            color: #757575;
            margin: 10px 0;
        """)
        self.pc_price.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pc_price)
        
        # ========== スペック詳細 ==========
        self.pc_specs = QLabel("分析を実行すると\nスペックが表示されます")
        self.pc_specs.setStyleSheet(f"""
            font-size: {FontSize.REC_SPECS}px; 
            color: #757575;
            background-color: white;
            padding: 6px;
            border-radius: 5px;
            border: 1px solid #E0E0E0;
        """)
        self.pc_specs.setWordWrap(True)
        self.pc_specs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pc_specs)
        
        # ========== スコア ==========
        self.match_score = QLabel("適合スコア: ―")
        self.match_score.setStyleSheet(f"""
            font-size: {FontSize.REC_SCORE}px; 
            font-weight: bold;
            color: #757575;
            background-color: #E3F2FD;
            padding: 6px;
            border-radius: 5px;
            margin-top: 5px;
        """)
        self.match_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.match_score)
        
        # ========== 区切り線 ==========
        layout.addSpacing(6)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #BDBDBD;")
        layout.addWidget(separator2)
        layout.addSpacing(3)
        
        # ========== 現在のプリセット ==========
        preset_label = QLabel("【選択中のプリセット】")
        preset_label.setStyleSheet(f"font-size: {FontSize.REC_PRESET_LABEL}px; font-weight: bold; color: #757575;")
        layout.addWidget(preset_label)
        
        self.current_preset = QLabel("一般ユーザー")
        self.current_preset.setStyleSheet(f"""
            font-size: {FontSize.REC_PRESET}px; 
            font-weight: bold; 
            color: white;
            background-color: #388E3C;
            padding: 4px;
            border-radius: 5px;
        """)
        self.current_preset.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_preset)
        
        self.weight_info = QLabel("PC1=0.00, PC2=0.00")
        self.weight_info.setStyleSheet(f"""
            font-size: {FontSize.REC_WEIGHT}px; 
            color: #757575;
            margin-top: 2px;
        """)
        self.weight_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.weight_info)
        
        self.preset_desc = QLabel("")
        self.preset_desc.setStyleSheet(f"""
            font-size: {FontSize.REC_SUBTITLE}px; 
            color: #616161;
            font-style: italic;
            margin-top: 2px;
        """)
        self.preset_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preset_desc.setWordWrap(True)
        layout.addWidget(self.preset_desc)
        
        # 下部の余白
        layout.addStretch()
    
    def update_recommendation(self, best_pc, preset_name, w_pc2):
        """推奨PC情報を更新"""
        self.pc_name.setText(best_pc['model'])
        self.pc_name.setStyleSheet(f"""
            font-size: {FontSize.REC_PC_NAME}px; 
            font-weight: bold; 
            color: #212121;
            background-color: #F1F8E9;
            padding: 10px;
            border-radius: 8px;
            border: 3px solid #FFD700;
        """)
        
        self.pc_price.setText(f"¥{best_pc['price']:,.0f}")
        self.pc_price.setStyleSheet(f"""
            font-size: {FontSize.REC_PRICE}px; 
            font-weight: bold; 
            color: #FF6F00;
            margin: 10px 0;
        """)
        
        specs_text = f"""CPU: {best_pc['cpu_score']:.0f}
GPU: {best_pc['gpu_score']:.0f}
RAM: {best_pc['ram_gb']:.0f} GB
SSD: {best_pc['storage_gb']:.0f} GB
総合性能: {best_pc['total_perf']:.2f}"""
        self.pc_specs.setText(specs_text.strip())
        self.pc_specs.setStyleSheet(f"""
            font-size: {FontSize.REC_SPECS}px; 
            color: #616161;
            background-color: white;
            padding: 6px;
            border-radius: 5px;
            border: 1px solid #E0E0E0;
        """)
        self.pc_specs.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.match_score.setText(f"適合スコア: {best_pc['score']:.2f}")
        self.match_score.setStyleSheet(f"""
            font-size: {FontSize.REC_SCORE}px; 
            font-weight: bold;
            color: #1976D2;
            background-color: #E3F2FD;
            padding: 6px;
            border-radius: 5px;
            margin-top: 5px;
        """)
        
        self.current_preset.setText(preset_name)
        self.weight_info.setText(f"構成バランス={w_pc2:.2f}")
        
        # プリセットの説明を表示
        if preset_name in PRESETS:
            self.preset_desc.setText(PRESETS[preset_name]["description"])
        else:
            self.preset_desc.setText("カスタム設定")


# ================================
# CSV 管理タブ（分析用データの唯一の入力元）
# ================================

class CSVManager(QWidget):
    def __init__(self):
        super().__init__()
        self.current_csv_path = None
        layout = QVBoxLayout(self)
        
        self.fields = [
            ("model", "モデル名"),
            ("cpu_score", "CPUスコア"),
            ("gpu_score", "GPUスコア"),
            ("ram_gb", "RAM (GB)"),
            ("storage_gb", "Storage (GB)"),
            ("price", "価格 (円)")
        ]
        
        self.headers = [f[0] for f in self.fields]
        self.inputs = {}
        
        # フォーム入力エリア
        form = QHBoxLayout()
        for key, label in self.fields:
            box = QVBoxLayout()
            box.addWidget(QLabel(label))
            edit = QLineEdit()
            self.inputs[key] = edit
            box.addWidget(edit)
            form.addLayout(box)
        
        layout.addLayout(form)
        
        # ボタンエリア
        btns = QHBoxLayout()
        
        add_btn = QPushButton("行を追加")
        add_btn.clicked.connect(self.add_row)
        btns.addWidget(add_btn)
        
        del_btn = QPushButton("選択行削除")
        del_btn.clicked.connect(self.delete_row)
        btns.addWidget(del_btn)
        
        new_btn = QPushButton("新規CSV保存")
        new_btn.clicked.connect(self.save_new_csv)
        btns.addWidget(new_btn)
        
        load_btn = QPushButton("既存CSV読込")
        load_btn.clicked.connect(self.load_existing_csv)
        btns.addWidget(load_btn)
        
        save_btn = QPushButton("変更を保存")
        save_btn.clicked.connect(self.save_existing_csv)
        btns.addWidget(save_btn)
        
        clear_btn = QPushButton("全消去")
        clear_btn.clicked.connect(self.clear_all)
        btns.addWidget(clear_btn)
        
        layout.addLayout(btns)
        
        # テーブル表示
        self.table = QTableWidget(0, len(self.fields))
        self.table.setHorizontalHeaderLabels(self.headers)
        layout.addWidget(self.table)
    
    def add_row(self):
        """フォームから行を追加"""
        # モデル名チェック
        if not self.inputs["model"].text().strip():
            QMessageBox.warning(self, "入力エラー", "モデル名を入力してください")
            return

        # 数値チェック
        for key in ["cpu_score", "gpu_score", "ram_gb", "storage_gb", "price"]:
            try:
                val = float(self.inputs[key].text())
                if val < 0:
                    QMessageBox.warning(self, "入力エラー", f"{key} は正の数で入力してください")
                    return
            except ValueError:
                QMessageBox.warning(self, "入力エラー", f"{key} は数値で入力してください")
                return
        
        # 行追加
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, key in enumerate(self.headers):
            self.table.setItem(r, c, QTableWidgetItem(self.inputs[key].text().strip()))
        
        # フォームをクリア
        for key in self.inputs:
            self.inputs[key].clear()
    
    def delete_row(self):
        """選択行を削除"""
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            
    def clear_all(self):
        """全行を削除"""
        if QMessageBox.question(self, "確認", "全てのデータを消去しますか？") == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)
    
    def _collect_models(self):
        """テーブル内のモデル名をリスト化"""
        models = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            models.append(item.text() if item else "")
        return models
    
    def _check_duplicates(self):
        """重複モデル名をチェック"""
        models = self._collect_models()
        dup = {m for m in models if models.count(m) > 1}
        if dup:
            QMessageBox.warning(self, "重複エラー", f"重複モデルがあります: {', '.join(dup)}")
            return False
        return True
    
    def _check_data_validity(self):
        """データの妥当性をチェック"""
        numeric_cols = [1, 2, 3, 4, 5]  # cpu, gpu, ram, storage, price
        for r in range(self.table.rowCount()):
            for c in numeric_cols:
                item = self.table.item(r, c)
                val_str = item.text() if item else ""
                try:
                    val = float(val_str)
                    if val < 0:
                        QMessageBox.warning(self, "エラー", f"行{r+1}, 列{c+1}：正の値を入力してください")
                        return False
                    if c == 5 and val == 0: # 価格は0不可
                        QMessageBox.warning(self, "エラー", f"行{r+1}：価格は0より大きい値を入力してください")
                        return False
                except ValueError:
                    QMessageBox.warning(self, "エラー", f"行{r+1}, 列{c+1}：数値が不正です ({val_str})")
                    return False
        return True
    
    def save_new_csv(self):
        """新規CSVを保存"""
        if not self._check_duplicates():
            return
        if not self._check_data_validity():
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "CSV保存", "pc_data.csv", "CSV (*.csv)")
        if not path:
            return
        
        self._write_csv(path)
        self.current_csv_path = path
        
        with open(LAST_CSV_FILE, "w", encoding="utf-8") as f:
            f.write(path)
        
        QMessageBox.information(self, "保存完了", "CSVを保存しました")
    
    def load_existing_csv(self):
        """既存CSVを読み込み"""
        path, _ = QFileDialog.getOpenFileName(self, "CSV読込", "", "CSV (*.csv)")
        if not path:
            return
        
        if self.load_csv_to_table(path):
            self.current_csv_path = path
            QMessageBox.information(self, "読込完了", "CSVを読み込みました")

    def load_csv_to_table(self, path):
        """CSVファイルを読み込んでテーブルに表示するヘルパーメソッド"""
        try:
            df = pd.read_csv(path)
            self.table.setRowCount(0)
            for _, row in df.iterrows():
                r = self.table.rowCount()
                self.table.insertRow(r)
                for c, h in enumerate(self.headers):
                    val = row[h] if h in row else ""
                    self.table.setItem(r, c, QTableWidgetItem(str(val)))
            return True
        except Exception as e:
            QMessageBox.critical(self, "読込エラー", f"CSVを読み込めません: {e}")
            return False
    
    def save_existing_csv(self):
        """編集済みのCSVを保存"""
        if not self.current_csv_path:
            QMessageBox.warning(self, "注意", "既存CSVが読み込まれていません")
            return
        
        if not self._check_duplicates():
            return
        if not self._check_data_validity():
            return
        
        self._write_csv(self.current_csv_path)
        QMessageBox.information(self, "保存完了", "変更を保存しました")
    
    def _write_csv(self, path):
        """CSVをファイルに書き込み"""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for r in range(self.table.rowCount()):
                row_data = []
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)
    
    def get_dataframe(self):
        """テーブル内容をDataFrameに変換"""
        if self.table.rowCount() == 0:
            return None
        
        data = []
        for r in range(self.table.rowCount()):
            row = {}
            for c, h in enumerate(self.headers):
                item = self.table.item(r, c)
                row[h] = item.text() if item else ""
            data.append(row)
        
        df = pd.DataFrame(data)
        for col in ["cpu_score", "gpu_score", "ram_gb", "storage_gb", "price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df


# ================================
# メインアプリ
# ================================

class PCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCコスパ分析 統合アプリ")
        self.resize(1440, 810)
        
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        
        self.csv_tab = CSVManager()
        tabs.addTab(self.csv_tab, "CSV管理")
        
        self.analysis_tab = QWidget()
        tabs.addTab(self.analysis_tab, "コスパ分析")
        
        self.current_preset_name = "一般ユーザー"
        self.init_analysis_tab()
        self.load_last_csv()
    
    def init_analysis_tab(self):
        """分析タブの初期化（3カラムレイアウト）"""
        main_layout = QVBoxLayout(self.analysis_tab)
        
        # ========== 上部：ボタン ==========
        top_layout = QHBoxLayout()
        analyze_btn = QPushButton("このデータで分析")
        analyze_btn.setMinimumHeight(32)
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2196F3;
                color: white;
                font-size: {FontSize.BTN_MAIN}px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 15px;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
        """)
        analyze_btn.clicked.connect(self.analyze_from_manager)
        top_layout.addWidget(analyze_btn)
        
        reload_btn = QPushButton("CSVを再読込")
        reload_btn.setMinimumHeight(32)
        reload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #FF9800;
                color: white;
                font-size: {FontSize.BTN_MAIN}px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px 15px;
            }}
            QPushButton:hover {{
                background-color: #F57C00;
            }}
        """)
        reload_btn.clicked.connect(self.reload_csv)
        top_layout.addWidget(reload_btn)
        
        top_layout.addStretch()
        main_layout.addLayout(top_layout)
        
        # ========== 中央：3カラム ==========
        content_layout = QHBoxLayout()
        
        # 左パネル（PCA情報）
        self.pca_panel = PCAInfoPanel()
        content_layout.addWidget(self.pca_panel, 2)  # 20%
        
        # 中央パネル（グラフ）
        if HAS_MATPLOTLIB:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            self.canvas = FigureCanvas(self.fig)
            self.canvas.mpl_connect("button_press_event", self.on_point_click)
            content_layout.addWidget(self.canvas, 5)  # 50%
        else:
            no_plot_label = QLabel("matplotlib未インストールのため可視化不可")
            no_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_layout.addWidget(no_plot_label, 5)
        
        # 右パネル（推奨PC）
        self.recommendation_panel = RecommendationPanel()
        content_layout.addWidget(self.recommendation_panel, 3)  # 30%
        
        main_layout.addLayout(content_layout)
        
        # ========== 下部：プリセット選択 ==========
        preset_container = QVBoxLayout()
        preset_label = QLabel("【プリセット選択】")
        preset_label.setStyleSheet(f"font-size: {FontSize.PRESET_LABEL}px; font-weight: bold; color: #424242; margin-top: 10px;")
        preset_container.addWidget(preset_label)
        
        preset_layout = QHBoxLayout()
        self.preset_buttons = {}
        
        for name, preset in PRESETS.items():
            btn = QPushButton(name)
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(90)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {preset['color']};
                    color: white;
                    font-size: {FontSize.BTN_PRESET}px;
                    font-weight: bold;
                    border-radius: 8px;
                    border: 2px solid {preset['color']};
                    padding: 6px 10px;
                }}
                QPushButton:hover {{
                    background-color: white;
                    color: {preset['color']};
                    border: 3px solid {preset['color']};
                }}
                QPushButton:pressed {{
                    background-color: #E0E0E0;
                }}
            """)
            btn.clicked.connect(lambda checked, n=name: self.apply_preset(n))
            self.preset_buttons[name] = btn
            preset_layout.addWidget(btn)
        
        preset_container.addLayout(preset_layout)
        main_layout.addLayout(preset_container)
        
        # ========== 下部：スライダー ==========
        slider_container = QVBoxLayout()
        slider_container.addSpacing(15)
        
        # 構成バランススライダー（データ読込後に動的ラベルが設定される）
        w_pc2_layout = QHBoxLayout()
        self.w_pc2_label = QLabel("PC2: 0% (データ読込後に表示)")
        self.w_pc2_label.setStyleSheet(f"font-size: {FontSize.SLIDER_LABEL}px; font-weight: bold; color: #2196F3; min-width: 250px;")
        w_pc2_layout.addWidget(self.w_pc2_label)
        
        self.w_pc2 = QSlider(Qt.Orientation.Horizontal)
        self.w_pc2.setRange(-100, 100)
        self.w_pc2.setValue(0)
        self.w_pc2.setMinimumWidth(400)
        self.w_pc2.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0.5, y2:0, x3:1, y3:0, stop:0 #7B1FA2, stop:0.5 #E0E0E0, stop:1 #2196F3);
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 2px solid #1565C0;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
        """)
        self.w_pc2.valueChanged.connect(self.on_weight_changed)
        w_pc2_layout.addWidget(self.w_pc2)
        w_pc2_layout.addStretch()
        
        slider_container.addLayout(w_pc2_layout)

        # 価格フィルタースライダー
        price_filter_layout = QHBoxLayout()
        self.price_label = QLabel("予算上限: 無制限")
        self.price_label.setStyleSheet(f"font-size: {FontSize.SLIDER_LABEL}px; font-weight: bold; color: #FF6F00; min-width: 200px;")
        price_filter_layout.addWidget(self.price_label)

        self.price_slider = QSlider(Qt.Orientation.Horizontal)
        self.price_slider.setRange(5, 100) # 5万〜100万
        self.price_slider.setValue(100)
        self.price_slider.setMinimumWidth(400)
        self.price_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 10px;
                background: #FFE0B2;
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #FF6F00;
                border: 2px solid #E65100;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
        """)
        self.price_slider.valueChanged.connect(self.on_weight_changed)
        price_filter_layout.addWidget(self.price_slider)
        price_filter_layout.addStretch()

        slider_container.addLayout(price_filter_layout)
        main_layout.addLayout(slider_container)
    
    def on_weight_changed(self, value):
        """スライダーが変更された時の共通処理"""
        # 動的に生成されたPC2の軸ラベルを使用（データ読込前はデフォルト値）
        pc2_desc = getattr(self, "pc2_desc", "構成バランス")
        
        # PC2スライダーの表示（動的ラベルを使用）
        balance_val = self.w_pc2.value()
        # 軸ラベルから左側と右側の意味を抽出（"左 ↔ 右"の形式を想定）
        if "↔" in pc2_desc:
            parts = pc2_desc.split("↔")
            left_label = parts[0].strip()
            right_label = parts[1].strip()
            
            if balance_val > 20:
                direction = f"→ {right_label}"
            elif balance_val < -20:
                direction = f"{left_label} ←"
            else:
                direction = "中間"
            self.w_pc2_label.setText(f"PC2: {balance_val}% ({direction})")
        else:
            # ↔が含まれない場合（データ読込前など）
            self.w_pc2_label.setText(f"PC2: {balance_val}%")
        
        p_val = self.price_slider.value()
        if p_val == 100:
            self.price_label.setText("予算上限: 無制限")
        else:
            self.price_label.setText(f"予算上限: {p_val}万円")

        # 手動操作時はプリセット選択を解除
        if not self.signals_blocked():
            self.current_preset_name = "カスタム"
        
        # PCAは再実行せず、スコア計算と描画のみ更新
        if hasattr(self, "df"):
            self._calculate_scores_and_pareto()
            self._update_visualization()
            self._update_info_panels()

    def signals_blocked(self):
        return self.w_pc2.signalsBlocked() or self.price_slider.signalsBlocked()
    
    def apply_preset(self, preset_name):
        """プリセット選択時の処理"""
        preset = PRESETS[preset_name]
        self.current_preset_name = preset_name
        
        # スライダーを更新（シグナルを一時停止して無限ループを防ぐ）
        self.w_pc2.blockSignals(True)
        self.price_slider.blockSignals(True)
        
        self.w_pc2.setValue(preset["w_pc2"])
        self.price_slider.setValue(100) # プリセット時は予算リセット
        
        self.w_pc2.blockSignals(False)
        self.price_slider.blockSignals(False)
        
        # ラベル更新と分析結果の更新（PCAは再実行しない）
        self.on_weight_changed(0)
    
    def reload_csv(self):
        """CSVタブのデータを再読込"""
        if self.csv_tab.current_csv_path and os.path.exists(self.csv_tab.current_csv_path):
            if self.csv_tab.load_csv_to_table(self.csv_tab.current_csv_path):
                QMessageBox.information(self, "再読込完了", "CSVを再読込しました")
                self.analyze_from_manager()
        else:
            QMessageBox.warning(self, "警告", "読み込むCSVファイルが見つかりません")
    
    
    def analyze_from_manager(self):
        """CSV管理タブのデータで分析を実行"""
        if not self.csv_tab._check_duplicates():
            return
        if not self.csv_tab._check_data_validity():
            return
            
        df = self.csv_tab.get_dataframe()
        if df is None or len(df) < 2:
            QMessageBox.warning(self, "警告", "分析には少なくとも2台以上のPCデータが必要です")
            return
        
        try:
            self.df = df.copy()
            self._run_pca()
            self._calculate_scores_and_pareto()
            self._update_visualization()
            self._update_info_panels()
        except Exception as e:
            QMessageBox.critical(self, "分析エラー", f"分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    def _generate_dynamic_label(self, component_vector, feature_names):
        """
        主成分の固有ベクトルから動的にラベルを生成する
        
        Args:
            component_vector: PCAの固有ベクトル（1つの主成分）
            feature_names: 特徴量の名前リスト
        
        Returns:
            str: 動的に生成された軸ラベル
        """
        # 寄与度の絶対値が大きい順にインデックスを取得
        sorted_indices = np.argsort(np.abs(component_vector))[::-1]
        
        pos_features = []  # プラスに寄与している特徴量
        neg_features = []  # マイナスに寄与している特徴量
        
        # しきい値（これより小さい寄与は無視）
        threshold = 0.2
        
        for idx in sorted_indices:
            val = component_vector[idx]
            if abs(val) < threshold:
                continue
            
            if val > 0:
                pos_features.append(feature_names[idx])
            else:
                neg_features.append(feature_names[idx])
        
        # ラベル生成ロジック
        # パターンA: プラスとマイナス両方に強い要素がある（対立軸）
        if pos_features and neg_features:
            pos_str = "・".join(pos_features[:2])  # 長くなりすぎないよう上位2つまで
            neg_str = "・".join(neg_features[:2])
            return f"{neg_str}重視 ↔ {pos_str}重視"
        
        # パターンB: 全部プラス（総合力軸）
        elif pos_features:
            return f"ロースペック ↔ {pos_features[0]}等ハイスペック"
        
        # パターンC: 全部マイナス（総合力軸の逆）
        elif neg_features:
            return f"{neg_features[0]}等ハイスペック ↔ ロースペック"
        
        return "特徴なし"

    def _run_pca(self):
        """PCA（主成分分析）の実行：総合性能と構成バランスを抽出"""
        features = ["cpu_score", "gpu_score", "ram_gb", "storage_gb"]
        feature_display_names = ["CPU", "GPU", "RAM", "SSD"]
        X = self.df[features].values
        
        # 1. 標準化（各特徴量のスケールを揃える）
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. PCA実行（行中心化なし：素直にデータの分散を見る）
        # PC1: 通常は全特徴量の総合力（総合性能）
        # PC2: 特徴量間の対立軸（例：GPU重視 vs CPU重視）
        n_comp = min(2, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp)
        pcs = self.pca.fit_transform(X_scaled)
        
        self.df["PC1"] = pcs[:, 0]
        self.df["PC2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0
        
        # 3. 総合性能（サイズ用）と価格（色用）の準備
        # 総合性能は標準化後の値の平均を使用（スケールを揃えた上で平均）
        self.df["total_perf"] = X_scaled.mean(axis=1)
        self.df["price_norm"] = (self.df["price"] - self.df["price"].min()) / (self.df["price"].max() - self.df["price"].min() + 1e-9)

        # 4. 軸の意味を動的に判定（データから自動生成）
        self.pc1_desc = self._generate_dynamic_label(
            self.pca.components_[0], 
            feature_display_names
        )
        if self.pca.components_.shape[0] >= 2:
            self.pc2_desc = self._generate_dynamic_label(
                self.pca.components_[1], 
                feature_display_names
            )
        else:
            self.pc2_desc = "なし"

    def _calculate_scores_and_pareto(self):
        """嗜好ベクトルによる推薦スコアの計算"""
        w_pc2 = self.w_pc2.value() / 100.0
        
        # 予算フィルター
        max_price = self.price_slider.value() * 10000
        if self.price_slider.value() == 100:
            max_price = float('inf')
        
        self.df["is_affordable"] = self.df["price"] <= max_price
        
        # スコア計算用のデータセット（予算内があればそれ、なければ全体）
        score_df = self.df[self.df["is_affordable"]] if self.df["is_affordable"].any() else self.df
        
        # PC1とPC2を正規化（同じスケールで評価するため）
        pc1_min = score_df["PC1"].min()
        pc1_max = score_df["PC1"].max()
        if pc1_max - pc1_min > 1e-9:
            pc1_norm = (score_df["PC1"] - pc1_min) / (pc1_max - pc1_min)
        else:
            pc1_norm = 0.5
        
        pc2_min = score_df["PC2"].min()
        pc2_max = score_df["PC2"].max()
        if pc2_max - pc2_min > 1e-9:
            pc2_norm = (score_df["PC2"] - pc2_min) / (pc2_max - pc2_min)  # 0-1に正規化
            # w_pc2が-1～+1の範囲なので、-1～+1にスケール
            pc2_scaled = (pc2_norm - 0.5) * 2  # -1～+1の範囲に変換
        else:
            pc2_scaled = 0
        
        # スコア = 性能(50%) + 構成の好み(50%)
        # 両方とも-1～+1の範囲で同等に評価
        self.df.loc[score_df.index, "score"] = 0.5 * (pc1_norm - 0.5) * 2 + 0.5 * w_pc2 * pc2_scaled
        
        # 最高スコアのPCを選択（予算内優先）
        if self.df["is_affordable"].any():
            self.best_pc = self.df[self.df["is_affordable"]].sort_values("score", ascending=False).iloc[0]
        else:
            self.best_pc = self.df.sort_values("score", ascending=False).iloc[0]

    def _update_visualization(self):
        """グラフの更新：PCA空間（構成の偏り）を可視化"""
        if not HAS_MATPLOTLIB or not hasattr(self, "pca"):
            return
            
        # 現在の表示範囲を保持（スライダー操作で表示が飛ばないようにする）
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.clear()
        
        # 散布図の描画
        # 色：価格（安いほど明るい/高いほど暗い）
        # サイズ：総合性能（大きいほど高性能）
        # total_perfを正規化して適切なサイズ範囲（20-100）にマッピング
        perf_min = self.df["total_perf"].min()
        perf_max = self.df["total_perf"].max()
        if perf_max - perf_min > 1e-9:
            perf_norm = (self.df["total_perf"] - perf_min) / (perf_max - perf_min)
        else:
            perf_norm = 0.5
        sizes = 20 + perf_norm * 80  # 20から100の範囲
        
        scatter = self.ax.scatter(
            self.df["PC1"], self.df["PC2"],
            c=self.df["price"], cmap="viridis_r",
            s=sizes,
            alpha=0.6, edgecolors="white", linewidth=0.5, label="PCモデル"
        )
        
        # 予算外のPCをグレーアウト
        out_of_budget = self.df[~self.df["is_affordable"]]
        if not out_of_budget.empty:
            out_perf_norm = (out_of_budget["total_perf"] - perf_min) / (perf_max - perf_min + 1e-9)
            out_sizes = 20 + out_perf_norm * 80
            self.ax.scatter(
                out_of_budget["PC1"], out_of_budget["PC2"],
                c="lightgray", s=out_sizes,
                alpha=0.3, edgecolors="none", zorder=2
            )

        # 推奨PCを強調
        best_perf_norm = (self.best_pc["total_perf"] - perf_min) / (perf_max - perf_min + 1e-9)
        best_size = 50 + best_perf_norm * 150  # 50から200の範囲（目立たせる）
        self.ax.scatter(
            self.best_pc["PC1"], self.best_pc["PC2"],
            c="red", s=best_size,
            marker="*", edgecolors="yellow", linewidth=1.5, zorder=10, label="推奨PC"
        )
        
        # 軸ラベルとタイトルの設定
        self.ax.set_xlabel(self.pc1_desc, fontsize=FontSize.GRAPH_AXIS, fontweight='bold')
        self.ax.set_ylabel(self.pc2_desc, fontsize=FontSize.GRAPH_AXIS, fontweight='bold')
        self.ax.set_title("PC構成分析 (PCA空間)", fontsize=FontSize.GRAPH_TITLE, fontweight='bold')
        
        # カラーバーの更新（既存の軸を再利用して「つぶれ」を防止）
        if not hasattr(self, "cax"):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes("right", size="5%", pad=0.1)
            self.colorbar = self.fig.colorbar(scatter, cax=self.cax, label="価格 (円)")
        else:
            # 2回目以降は既存のカラーバーを更新
            self.colorbar.update_normal(scatter)
            
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc='best', fontsize=FontSize.GRAPH_LEGEND)
        
        # 範囲の固定（同じ分析結果を表示している間のみ）
        if hasattr(self, "_prev_pca_id") and self._prev_pca_id == id(self.pca):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self._prev_pca_id = id(self.pca)
            self.fig.tight_layout()
        
        self.canvas.draw()

    def _update_info_panels(self):
        """左右のパネルを更新"""
        # 左パネル
        self.pca_panel.update_pca_info(
            self.pca, 
            self.pca.explained_variance_ratio_,
            pc1_desc=self.pc1_desc,
            pc2_desc=self.pc2_desc
        )
        
        # 右パネル
        self.recommendation_panel.update_recommendation(
            best_pc=self.best_pc,
            preset_name=self.current_preset_name,
            w_pc2=self.w_pc2.value() / 100.0
        )
        
        # プリセットボタンのハイライト更新
        self._update_preset_button_styles()

    def _update_preset_button_styles(self):
        """選択中のプリセットボタンを強調表示"""
        for name, btn in self.preset_buttons.items():
            preset = PRESETS[name]
            if name == self.current_preset_name:
                # 選択中：背景を白、文字をプリセット色、太い枠線
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: white;
                        color: {preset['color']};
                        font-size: {FontSize.BTN_PRESET}px;
                        font-weight: bold;
                        border-radius: 8px;
                        border: 4px solid {preset['color']};
                        padding: 6px 10px;
                    }}
                """)
            else:
                # 非選択：通常スタイル
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {preset['color']};
                        color: white;
                        font-size: {FontSize.BTN_PRESET}px;
                        font-weight: bold;
                        border-radius: 8px;
                        border: 2px solid {preset['color']};
                        padding: 6px 10px;
                    }}
                    QPushButton:hover {{
                        background-color: white;
                        color: {preset['color']};
                        border: 3px solid {preset['color']};
                    }}
                """)

    def on_point_click(self, event):
        """グラフ上の点をクリックしてモデル詳細を表示"""
        if not hasattr(self, "df") or event.inaxes != self.ax:
            return
        
        # PCA空間で最も近い点を探す
        dists = (self.df["PC1"] - event.xdata)**2 + (self.df["PC2"] - event.ydata)**2
        idx = dists.idxmin()
        row = self.df.loc[idx]
        
        # 推奨PCかどうかを判定
        is_best = False
        if hasattr(self, 'best_pc'):
            is_best = (row['model'] == self.best_pc['model'])
        
        best_mark = "⭐ " if is_best else ""
        budget_status = "予算内" if row['is_affordable'] else "予算外"
        
        QMessageBox.information(
            self,
            f"{best_mark}モデル詳細",
            f"モデル: {row['model']}\n"
            f"価格: {row['price']:,.0f} 円 ({budget_status})\n"
            f"CPUスコア: {row['cpu_score']:.0f}\n"
            f"GPUスコア: {row['gpu_score']:.0f}\n"
            f"RAM: {row['ram_gb']:.1f} GB\n"
            f"SSD: {row['storage_gb']:.0f} GB\n"
            f"総合性能: {row['total_perf']:.2f}\n"
            f"適合スコア: {row['score']:.2f}"
            + (f"\n\n⭐ あなたへの推奨PC" if is_best else "")
        )
    
    def load_last_csv(self):
        """前回使用したCSVを自動読み込み"""
        if os.path.exists(LAST_CSV_FILE):
            with open(LAST_CSV_FILE, "r", encoding="utf-8") as f:
                path = f.read().strip()
            
            if path and os.path.exists(path):
                if self.csv_tab.load_csv_to_table(path):
                    self.csv_tab.current_csv_path = path
                    # 初回起動時も分析を実行
                    self.analyze_from_manager()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCApp()
    win.show()
    sys.exit(app.exec())
