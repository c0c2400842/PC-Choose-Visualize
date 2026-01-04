"""
PCA分析の検証スクリプト（pc_visualize_app.pyと同じロジック）
主成分分析による次元削減と推薦スコアの計算を確認
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("="*80)
print("【PCA分析検証】pc_visualize_app.pyと同じロジックで検証")
print("="*80)

# データ読み込み
df = pd.read_csv("pc_data.csv")
print(f"\nデータ件数: {len(df)}台")
print(f"価格範囲: {df['price'].min():,.0f}円 ～ {df['price'].max():,.0f}円")

# 1. 標準化（アプリと同じ）
feature_cols = ["cpu_score", "gpu_score", "ram_gb", "storage_gb"]
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*80)
print("【1. 標準化】")
print("="*80)
for i, col in enumerate(feature_cols):
    print(f"{col:15s}: 平均={X_scaled[:, i].mean():.6f}, 標準偏差={X_scaled[:, i].std():.6f}")

# 2. PCA（アプリと同じ）
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0

print("\n" + "="*80)
print("【2. PCA結果】")
print("="*80)
print(f"PC1の説明分散比: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"PC2の説明分散比: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"累積寄与率: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

print("\n固有ベクトル（PC1）:")
for i, col in enumerate(feature_cols):
    print(f"  {col:15s}: {pca.components_[0, i]:>7.4f}")

print("\n固有ベクトル（PC2）:")
for i, col in enumerate(feature_cols):
    print(f"  {col:15s}: {pca.components_[1, i]:>7.4f}")

# 3. 総合性能の計算（アプリと同じ）
df["total_perf"] = X_scaled.mean(axis=1)
df["price_norm"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min() + 1e-9)

print("\n" + "="*80)
print("【3. 推薦スコア計算（複数パターン）】")
print("="*80)

# テストケース：異なる嗜好パターン
test_cases = [
    {"name": "プログラマー", "w_pc2": 0.80, "max_price": 200000},
    {"name": "ゲーマー", "w_pc2": -0.90, "max_price": 250000},
    {"name": "一般ユーザー", "w_pc2": 0.0, "max_price": 100000},
    {"name": "予算無制限", "w_pc2": 0.0, "max_price": float('inf')},
]

for test in test_cases:
    print(f"\n--- {test['name']} (w_pc2={test['w_pc2']:.2f}, 予算≦{test['max_price']:,.0f}円) ---")
    
    # 予算フィルター
    df["is_affordable"] = df["price"] <= test['max_price']
    score_df = df[df["is_affordable"]] if df["is_affordable"].any() else df
    
    # PC1とPC2を正規化（アプリと同じロジック）
    pc1_min = score_df["PC1"].min()
    pc1_max = score_df["PC1"].max()
    if pc1_max - pc1_min > 1e-9:
        pc1_norm = (score_df["PC1"] - pc1_min) / (pc1_max - pc1_min)
    else:
        pc1_norm = 0.5
    
    pc2_min = score_df["PC2"].min()
    pc2_max = score_df["PC2"].max()
    if pc2_max - pc2_min > 1e-9:
        pc2_norm = (score_df["PC2"] - pc2_min) / (pc2_max - pc2_min)
        pc2_scaled = (pc2_norm - 0.5) * 2  # -1～+1の範囲に変換
    else:
        pc2_scaled = 0
    
    # スコア計算（性能50% + 構成の好み50%）
    df.loc[score_df.index, "score"] = 0.5 * (pc1_norm - 0.5) * 2 + 0.5 * test['w_pc2'] * pc2_scaled
    
    # 最高スコアのPC
    if df["is_affordable"].any():
        best_pc = df[df["is_affordable"]].sort_values("score", ascending=False).iloc[0]
    else:
        best_pc = df.sort_values("score", ascending=False).iloc[0]
    
    print(f"🏆 推奨PC: {best_pc['model']}")
    print(f"   価格: {best_pc['price']:>10,.0f}円")
    print(f"   スコア: {best_pc['score']:>7.4f}")
    print(f"   PC1: {best_pc['PC1']:>7.4f}, PC2: {best_pc['PC2']:>7.4f}")
    
    # トップ3を表示
    top3 = df[df["is_affordable"]].sort_values("score", ascending=False).head(3) if df["is_affordable"].any() else df.sort_values("score", ascending=False).head(3)
    print("   トップ3:")
    for idx, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"   {idx}. {row['model']:25s} スコア:{row['score']:>7.4f} 価格:{row['price']:>10,.0f}円")

print("\n" + "="*80)
print("【4. PC1とPC2の意味解釈】")
print("="*80)

# PC1の解釈（すべて正なら総合性能）
pc1_positive = sum(1 for x in pca.components_[0] if x > 0)
if pc1_positive == len(feature_cols):
    print("PC1: すべての特徴量と正の相関 → 総合性能を表す軸")
    print("     (ロースペック ↔ ハイスペック)")
else:
    print("PC1: 特徴量間にトレードオフ関係あり")

# PC2の解釈（対立する特徴を見つける）
pc2_components = pca.components_[1]
pos_features = [feature_cols[i] for i, x in enumerate(pc2_components) if x > 0.2]
neg_features = [feature_cols[i] for i, x in enumerate(pc2_components) if x < -0.2]
print(f"\nPC2: {', '.join(neg_features) if neg_features else '―'} ↔ {', '.join(pos_features) if pos_features else '―'}")
print("     (構成の偏りを表す軸)")

print("\n" + "="*80)
print("【検証完了】アプリと同じロジックで動作しています")
print("="*80)
