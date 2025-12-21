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
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

LAST_CSV_FILE = "last_csv_path.txt"

# プリセット定義（w1: 総合性能 0-100, w2: 特化方向 -100 to 100）
PRESETS = {
    "開発者向け": {
        "w1": 90, "w2": 60,
        "color": "#1976D2",  # 青
        "description": "CPU・RAM重視"
    },
    "ゲーマー向け": {
        "w1": 60, "w2": -80,
        "color": "#D32F2F",  # 赤
        "description": "GPU・ストレージ重視"
    },
    "一般ユーザー": {
        "w1": 70, "w2": 0,
        "color": "#388E3C",  # 緑
        "description": "バランス型"
    },
    "学生向け": {
        "w1": 50, "w2": -40,
        "color": "#FFA000",  # オレンジ
        "description": "コスパ重視"
    },
    "クリエイター": {
        "w1": 90, "w2": 30,
        "color": "#7B1FA2",  # 紫
        "description": "高性能重視"
    }
}

# ================================
# フォントサイズ設定（一箇所で管理）
# ================================
FONT_SCALE = 0.68  # フォントサイズの倍率（10%縮小: 0.75 -> 0.68）

class FontSize:
    """フォントサイズを一括管理するクラス"""
    # 左パネル（PCA情報）
    PCA_TITLE = int(18 * FONT_SCALE)
    PCA_LABEL = int(12 * FONT_SCALE)
    PCA_VALUE = int(24 * FONT_SCALE)
    PCA_CUMSUM = int(14 * FONT_SCALE)
    PCA_CONTRIB_TITLE = int(14 * FONT_SCALE)
    PCA_TABLE = 8 # PC1,PC2の表はサイズを維持
    
    # 右パネル（推奨PC）
    REC_TITLE = int(20 * FONT_SCALE)
    REC_PC_NAME = int(18 * FONT_SCALE)
    REC_PRICE = int(36 * FONT_SCALE)
    REC_SPECS = int(13 * FONT_SCALE)
    REC_SECTION_TITLE = int(16 * FONT_SCALE)
    REC_PARETO = int(18 * FONT_SCALE)
    REC_PRESET_LABEL = int(13 * FONT_SCALE)
    REC_PRESET = int(16 * FONT_SCALE)
    REC_WEIGHT = int(12 * FONT_SCALE)
    REC_IDEAL_NAME = int(15 * FONT_SCALE)
    REC_IDEAL_INFO = int(13 * FONT_SCALE)
    REC_IDEAL_SUBTITLE = int(11 * FONT_SCALE)
    
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
        self.explanation = QLabel("PC1: 総合スペックの高さ\nPC2: 特化方向")
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
        self.pc1_label = QLabel("PC1寄与率（総合性能）")
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
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.pc1_bar)
        
        # ========== PC2寄与率 ==========
        layout.addSpacing(8)
        
        self.pc2_label = QLabel("PC2寄与率（特化方向）")
        self.pc2_label.setStyleSheet(f"font-size: {FontSize.PCA_LABEL}px; font-weight: bold;")
        layout.addWidget(self.pc2_label)
        
        self.pc2_value = QLabel("0.0%")
        self.pc2_value.setStyleSheet(f"font-size: {FontSize.PCA_VALUE}px; color: #2196F3; font-weight: bold;")
        layout.addWidget(self.pc2_value)
        
        self.pc2_bar = QProgressBar()
        self.pc2_bar.setRange(0, 100)
        self.pc2_bar.setValue(10)
        self.pc2_bar.setTextVisible(False)
        self.pc2_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                background-color: #E0E0E0;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        layout.addWidget(self.pc2_bar)
        
        # ========== 累積寄与率 ==========
        layout.addSpacing(3)
        
        self.cumsum_label = QLabel("累積寄与率: 89.9%")
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
        self.pc1_label.setText(f"PC1寄与率（{pc1_desc}）")
        self.pc2_label.setText(f"PC2寄与率（{pc2_desc}）")
        
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
            if abs(pc1_val) > 0.5:
                pc1_item.setBackground(QColor("#C8E6C9"))
            self.contrib_table.setItem(i, 0, pc1_item)
            
            # PC2
            if len(var_ratio) >= 2:
                pc2_val = components[1, i]
                pc2_item = QTableWidgetItem(f"{pc2_val:+.3f}")
                pc2_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if abs(pc2_val) > 0.5:
                    pc2_item.setBackground(QColor("#BBDEFB"))
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
        title = QLabel("🏆 総合評価1位PC")
        title.setStyleSheet(f"""
            font-size: {FontSize.REC_TITLE}px; 
            font-weight: bold; 
            color: #FF6F00;
            margin-bottom: 5px;
        """)
        layout.addWidget(title)
        
        subtitle = QLabel("（グラフの★に最も近い実在PC）")
        subtitle.setStyleSheet(f"font-size: {FontSize.REC_IDEAL_SUBTITLE}px; color: #757575; margin-top: -5px; margin-bottom: 5px;")
        layout.addWidget(subtitle)
        
        ideal_desc = QLabel("※★(理想点) = 最高性能かつ最低価格の点")
        ideal_desc.setStyleSheet(f"font-size: {FontSize.REC_IDEAL_SUBTITLE}px; color: #1976D2; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(ideal_desc)
        
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
        
        # ========== 理想点からの距離 ==========
        self.ideal_distance = QLabel("理想点からの距離: ―")
        self.ideal_distance.setStyleSheet(f"""
            font-size: {FontSize.REC_IDEAL_INFO}px; 
            color: #757575;
            background-color: white;
            padding: 6px;
            border-radius: 3px;
            border: 1px solid #E0E0E0;
            margin-top: 3px;
        """)
        self.ideal_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ideal_distance)
        
        # ========== 区切り線 ==========
        layout.addSpacing(6)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #BDBDBD;")
        layout.addWidget(separator2)
        layout.addSpacing(3)
        
        # ========== パレート点数 ==========
        pareto_title = QLabel("📊 分析結果")
        pareto_title.setStyleSheet(f"font-size: {FontSize.REC_SECTION_TITLE}px; font-weight: bold;")
        layout.addWidget(pareto_title)
        
        self.pareto_count = QLabel("分析後に表示")
        self.pareto_count.setStyleSheet(f"""
            font-size: {FontSize.REC_PARETO}px; 
            font-weight: bold; 
            color: #757575;
            background-color: #E3F2FD;
            padding: 5px;
            border-radius: 5px;
            margin-top: 2px;
        """)
        self.pareto_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pareto_count)
        
        # ========== 現在のプリセット ==========
        layout.addSpacing(8)
        
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
        
        self.weight_info = QLabel("w1=0.60, w2=0.40")
        self.weight_info.setStyleSheet(f"""
            font-size: {FontSize.REC_WEIGHT}px; 
            color: #757575;
            margin-top: 2px;
        """)
        self.weight_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.weight_info)
        
        # 下部の余白
        layout.addStretch()
    
    def update_recommendation(self, ideal_pc, pareto_count, preset_name, w1, w2):
        """推奨PC情報を更新（総合評価1位PCのみ表示）"""
        # 総合評価1位PC
        self.pc_name.setText(ideal_pc['model'])
        self.pc_name.setStyleSheet(f"""
            font-size: {FontSize.REC_PC_NAME}px; 
            font-weight: bold; 
            color: #212121;
            background-color: #F1F8E9;
            padding: 10px;
            border-radius: 8px;
            border: 3px solid #FFD700;
        """)
        
        self.pc_price.setText(f"¥{ideal_pc['price']:,.0f}")
        self.pc_price.setStyleSheet(f"""
            font-size: {FontSize.REC_PRICE}px; 
            font-weight: bold; 
            color: #FF6F00;
            margin: 10px 0;
        """)
        
        specs_text = f"""CPU: {ideal_pc['cpu_score']:.0f}
GPU: {ideal_pc['gpu_score']:.0f}
RAM: {ideal_pc['ram_gb']:.0f} GB
Storage: {ideal_pc['storage_gb']:.0f} GB
性能スコア: {ideal_pc['perf']:.2f}"""
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
        
        # 理想点からの距離
        self.ideal_distance.setText(f"理想点からの距離: {ideal_pc['distance']:.4f}")
        self.ideal_distance.setStyleSheet(f"""
            font-size: {FontSize.REC_IDEAL_INFO}px; 
            color: #424242;
            background-color: white;
            padding: 6px;
            border-radius: 3px;
            border: 1px solid #E0E0E0;
            margin-top: 3px;
        """)
        self.ideal_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.pareto_count.setText(f"パレート最適解: {pareto_count}台")
        self.pareto_count.setStyleSheet(f"""
            font-size: {FontSize.REC_PARETO}px; 
            font-weight: bold; 
            color: #1976D2;
            background-color: #E3F2FD;
            padding: 5px;
            border-radius: 5px;
            margin-top: 2px;
        """)
        self.pareto_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.current_preset.setText(preset_name)
        self.weight_info.setText(f"w1={w1:.2f}, w2={w2:.2f}")


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
    
    def _collect_models(self):
        """テーブル内のモデル名をリスト化"""
        return [self.table.item(r, 0).text() for r in range(self.table.rowCount())]
    
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
        
        # PC1スライダー
        w1_layout = QHBoxLayout()
        self.w1_label = QLabel("総合性能重視: 70%")
        self.w1_label.setStyleSheet(f"font-size: {FontSize.SLIDER_LABEL}px; font-weight: bold; color: #1976D2; min-width: 150px;")
        w1_layout.addWidget(self.w1_label)
        
        self.w1 = QSlider(Qt.Orientation.Horizontal)
        self.w1.setRange(0, 100)
        self.w1.setValue(70)
        self.w1.setMinimumWidth(400)
        self.w1.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E0E0E0, stop:1 #1976D2);
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                border: 2px solid #0D47A1;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
        """)
        self.w1.valueChanged.connect(self.on_w1_changed)
        w1_layout.addWidget(self.w1)
        w1_layout.addStretch()
        
        slider_container.addLayout(w1_layout)
        
        # PC2スライダー
        w2_outer_layout = QVBoxLayout()
        
        w2_header = QHBoxLayout()
        self.w2_label = QLabel("特化方向の調整: 0%")
        self.w2_label.setStyleSheet(f"font-size: {FontSize.SLIDER_LABEL}px; font-weight: bold; color: #2196F3; min-width: 150px;")
        w2_header.addWidget(self.w2_label)
        
        w2_desc = QLabel("← 減少で左側を重視 | 増加で右側を重視 →")
        w2_desc.setStyleSheet(f"font-size: {FontSize.REC_IDEAL_SUBTITLE}px; color: #757575;")
        w2_header.addWidget(w2_desc)
        w2_header.addStretch()
        w2_outer_layout.addLayout(w2_header)

        w2_layout = QHBoxLayout()
        self.w2_left_label = QLabel("SSD重視")
        self.w2_left_label.setStyleSheet("color: #F44336; font-weight: bold;")
        w2_layout.addWidget(self.w2_left_label)
        
        self.w2 = QSlider(Qt.Orientation.Horizontal)
        self.w2.setRange(-100, 100)
        self.w2.setValue(0)
        self.w2.setMinimumWidth(400)
        self.w2.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0.5, y2:0, x3:1, y3:0, stop:0 #F44336, stop:0.5 #E0E0E0, stop:1 #2196F3);
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
        self.w2.valueChanged.connect(self.on_w2_changed)
        w2_layout.addWidget(self.w2)
        
        self.w2_right_label = QLabel("CPU重視")
        self.w2_right_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        w2_layout.addWidget(self.w2_right_label)
        
        reset_btn = QPushButton("重みをリセット")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        reset_btn.clicked.connect(lambda: self.apply_preset("一般ユーザー"))
        w2_layout.addWidget(reset_btn)
        
        w2_outer_layout.addLayout(w2_layout)
        slider_container.addLayout(w2_outer_layout)
        main_layout.addLayout(slider_container)
    
    def on_w1_changed(self, value):
        """PC1スライダーが変更された時の処理"""
        self.w1_label.setText(f"総合性能重視: {value}%")
        # 手動操作時はプリセット選択を解除
        self.current_preset_name = "カスタム"
        self.analyze_from_manager()
    
    def on_w2_changed(self, value):
        """PC2スライダーが変更された時の処理"""
        self.w2_label.setText(f"特化方向の調整: {value}%")
        # 手動操作時はプリセット選択を解除
        self.current_preset_name = "カスタム"
        self.analyze_from_manager()
    
    def apply_preset(self, preset_name):
        """プリセット選択時の処理"""
        preset = PRESETS[preset_name]
        
        # 現在のプリセット名を記録
        self.current_preset_name = preset_name
        
        # スライダーを更新
        self.w1.blockSignals(True)
        self.w2.blockSignals(True)
        self.w1.setValue(preset["w1"])
        self.w2.setValue(preset["w2"])
        self.w1_label.setText(f"総合性能重視: {preset['w1']}%")
        self.w2_label.setText(f"特化方向の調整: {preset['w2']}%")
        self.w1.blockSignals(False)
        self.w2.blockSignals(False)
        
        # 分析実行
        self.analyze_from_manager()
    
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

    def _run_pca(self):
        """PCA（主成分分析）の実行"""
        X = self.df[["cpu_score", "gpu_score", "ram_gb", "storage_gb"]].values
        X_scaled = StandardScaler().fit_transform(X)
        
        n_comp = min(2, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp)
        pcs = self.pca.fit_transform(X_scaled)
        
        self.df["PC1"] = pcs[:, 0]
        self.df["PC2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0
        self.df["price_log"] = np.log(self.df["price"])

    def _calculate_scores_and_pareto(self):
        """性能スコアの計算とパレート最適解の抽出"""
        w1, w2 = self.w1.value() / 100.0, self.w2.value() / 100.0
        self.df["perf"] = w1 * self.df["PC1"] + w2 * self.df["PC2"]
        
        # 理想点の定義（最高性能・最低価格）
        self.ideal_perf = self.df["perf"].max()
        self.ideal_price_log = self.df["price_log"].min()
        
        # パレート最適解の抽出
        pareto_indices = []
        for i in range(len(self.df)):
            is_dominated = False
            for j in range(len(self.df)):
                if i == j: continue
                # 性能が高く、かつ価格が安いものがあればドミネートされる
                if (self.df.iloc[j]["perf"] >= self.df.iloc[i]["perf"] and 
                    self.df.iloc[j]["price"] <= self.df.iloc[i]["price"]):
                    if (self.df.iloc[j]["perf"] > self.df.iloc[i]["perf"] or 
                        self.df.iloc[j]["price"] < self.df.iloc[i]["price"]):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_indices.append(i)
        
        self.df["is_pareto"] = False
        self.df.loc[self.df.index[pareto_indices], "is_pareto"] = True
        
        # 理想点からの距離計算（パレート解のみ）
        pareto_df = self.df[self.df["is_pareto"]].copy()
        perf_range = self.df["perf"].max() - self.df["perf"].min()
        price_log_range = self.df["price_log"].max() - self.df["price_log"].min()
        
        if perf_range > 0 and price_log_range > 0:
            # 正規化された距離（視覚的な距離に合わせるためprice_logを使用）
            pareto_df["norm_perf"] = (self.ideal_perf - pareto_df["perf"]) / perf_range
            pareto_df["norm_price"] = (pareto_df["price_log"] - self.ideal_price_log) / price_log_range
            pareto_df["distance"] = np.sqrt(pareto_df["norm_perf"]**2 + pareto_df["norm_price"]**2)
        else:
            pareto_df["distance"] = 0
            
        self.ideal_best = pareto_df.sort_values("distance").iloc[0]
        self.pareto_count = len(pareto_df)

    def _update_visualization(self):
        """グラフの更新"""
        if not HAS_MATPLOTLIB:
            return
            
        self.ax.clear()
        w1, w2 = self.w1.value() / 100.0, self.w2.value() / 100.0
        
        # 非パレート点
        non_pareto = self.df[~self.df["is_pareto"]]
        self.ax.scatter(non_pareto["perf"], non_pareto["price_log"],
                       c="lightgray", s=80, alpha=0.4, label="Other", zorder=1)
        
        # パレート点（総合評価1位以外）
        pareto_df = self.df[self.df["is_pareto"]]
        pareto_others = pareto_df[pareto_df['model'] != self.ideal_best['model']]
        if not pareto_others.empty:
            self.ax.scatter(pareto_others["perf"], pareto_others["price_log"],
                           c="#4CAF50", s=250, label="Pareto Front",
                           edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        
        # 総合評価1位PC
        self.ax.scatter(self.ideal_best["perf"], self.ideal_best["price_log"],
                       c="#8BC34A", s=400, label="Best PC",
                       edgecolors='#FFD700', linewidths=4, zorder=5, marker='o')
        
        # 理想点
        self.ax.scatter(self.ideal_perf, self.ideal_price_log,
                       marker="*", s=500, c="#FFD700",
                       label="Ideal Point", zorder=4, edgecolors='#FF6F00')
        
        self.ax.set_xlabel("Performance Score", fontsize=FontSize.GRAPH_AXIS, fontweight='bold')
        self.ax.set_ylabel("Price (log scale)", fontsize=FontSize.GRAPH_AXIS, fontweight='bold')
        self.ax.set_title(f"Cost-Performance Analysis (w1={w1:.2f}, w2={w2:.2f})",
                        fontsize=FontSize.GRAPH_TITLE, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper left', fontsize=FontSize.GRAPH_LEGEND)
        
        self.canvas.draw()

    def _update_info_panels(self):
        """左右のパネルを更新"""
        # PCAの各成分の意味を動的に判定
        features = ['CPU', 'GPU', 'RAM', 'SSD']
        components = self.pca.components_
        
        # PC1: 全て正なら「総合性能」、そうでなければ最大寄与
        if np.all(components[0] >= 0):
            pc1_desc = "総合スペック"
        else:
            max_idx = np.argmax(np.abs(components[0]))
            pc1_desc = f"{features[max_idx]}重視"
            
        # PC2: 正負の最大寄与を特定して「A vs B」とする
        if components.shape[0] >= 2:
            pos_idx = np.argmax(components[1])
            neg_idx = np.argmin(components[1])
            pc2_pos_name = features[pos_idx]
            pc2_neg_name = features[neg_idx]
            
            if components[1, pos_idx] > 0.3 and components[1, neg_idx] < -0.3:
                pc2_desc = f"{pc2_pos_name} vs {pc2_neg_name}"
            else:
                max_idx = np.argmax(np.abs(components[1]))
                pc2_desc = f"{features[max_idx]}特化"
        else:
            pc2_pos_name = "右側"
            pc2_neg_name = "左側"
            pc2_desc = "なし"

        # スライダーのラベルを更新
        self.w2_left_label.setText(f"{pc2_neg_name}重視")
        self.w2_right_label.setText(f"{pc2_pos_name}重視")

        # 左パネル
        self.pca_panel.update_pca_info(
            self.pca, 
            self.pca.explained_variance_ratio_,
            pc1_desc=pc1_desc,
            pc2_desc=pc2_desc
        )
        
        # 右パネル
        self.ideal_best_model = self.ideal_best['model']
        self.recommendation_panel.update_recommendation(
            ideal_pc=self.ideal_best,
            pareto_count=self.pareto_count,
            preset_name=self.current_preset_name,
            w1=self.w1.value() / 100.0,
            w2=self.w2.value() / 100.0
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
        
        # 理想点の近くをクリックしたかチェック
        if hasattr(self, 'ideal_perf') and hasattr(self, 'ideal_price_log'):
            ideal_dist = (self.ideal_perf - event.xdata)**2 + (self.ideal_price_log - event.ydata)**2
            # 理想点が最も近い場合（閾値を設定）
            if ideal_dist < 0.05:
                QMessageBox.information(
                    self,
                    "理想点",
                    "これは理想的な点（最高性能・最低価格）です。\n実在するPCではありません。"
                )
                return
        
        # まず全PCから対数空間で最も近い点を探す
        dists_all = (self.df["perf"] - event.xdata)**2 + (self.df["price_log"] - event.ydata)**2
        idx = dists_all.idxmin()
        row = self.df.loc[idx]
        
        # 総合評価1位かどうかを判定
        is_best = False
        if hasattr(self, 'ideal_best_model'):
            is_best = (row['model'] == self.ideal_best_model)
        
        best_mark = "🏆 " if is_best else ""
        pareto_status = "はい" if row['is_pareto'] else "いいえ"
        
        QMessageBox.information(
            self,
            f"{best_mark}モデル詳細",
            f"モデル: {row['model']}\n"
            f"価格: {row['price']:,.0f} 円\n"
            f"CPUスコア: {row['cpu_score']:.0f}\n"
            f"GPUスコア: {row['gpu_score']:.0f}\n"
            f"RAM: {row['ram_gb']:.1f} GB\n"
            f"Storage: {row['storage_gb']:.0f} GB\n"
            f"性能スコア: {row['perf']:.2f}\n"
            f"パレート最適: {pareto_status}"
            + (f"\n\n🏆 総合評価1位PC" if is_best else "")
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
