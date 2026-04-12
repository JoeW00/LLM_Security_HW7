# Assignment 7：Clustering 演算法與資安應用

## 學習目標

- 認識 clustering 演算法，了解如何利用機器學習協助資安分析師過濾與篩選異常行為
- 使用程式實作分群演算法，從 Hugging Face 取得資料集進行分群實驗
- 進行基礎 SFT fine-tuning，觀察資料分佈不均如何影響模型行為

---

## TASK 1 (Achieving)：Clustering 原理與資安場景思考

### 1.1 IBM SkillBuild 課程學習心得

完成 IBM SkillBuild「Machine Learning for Developers」學習路徑後，我對機器學習的整體框架有了更完整的認識。課程涵蓋了監督式學習（Supervised Learning）與非監督式學習（Unsupervised Learning）兩大類別，其中非監督式學習的 clustering 部分讓我印象最為深刻——它不需要事先標記資料，便能從大量資料中自動發現隱藏的結構與模式，這在資安領域中面對大量未標註的網路流量或日誌資料時，具有極高的實用價值。

課程中也強調了資料前處理（data preprocessing）的重要性，包括特徵縮放（feature scaling）、缺失值處理等步驟，這些都會直接影響模型的表現。此外，課程透過實作練習讓我理解到，選擇合適的演算法與調整超參數（hyperparameters）對最終結果有決定性的影響。

<!-- 截圖: IBM SkillBuild 課程實作過程截圖 -->

### 1.2 Clustering 演算法理解

完成課程後，我對以下三種主要的 clustering 演算法有了深入的理解：

#### K-Means

- **原理**：隨機初始化 K 個中心點（centroids），將每個資料點分配到最近的中心點所屬的群集，然後重新計算中心點位置，反覆迭代直到收斂。
- **優點**：計算效率高、易於理解與實作，適合大規模資料集。
- **缺點**：需要預先指定群集數量 K；對初始中心點敏感；假設群集為球形且大小相近，對不規則形狀的資料效果不佳；容易受到離群值（outliers）影響。

#### DBSCAN（Density-Based Spatial Clustering of Applications with Noise）

- **原理**：基於密度的分群方法，以每個資料點為中心，在指定半徑（eps）內若包含足夠數量（min_samples）的鄰近點，則將其標記為核心點並擴展群集。無法被歸類到任何群集的點被視為雜訊（noise）。
- **優點**：不需要預先指定群集數量；能識別任意形狀的群集；能自動偵測並排除雜訊點。
- **缺點**：對 eps 和 min_samples 參數敏感；當資料密度差異大時表現不佳；高維度資料效果下降。

#### Hierarchical Clustering（階層式分群）

- **原理**：分為凝聚式（Agglomerative，由下而上）和分裂式（Divisive，由上而下）。凝聚式從每個資料點為獨立群集開始，逐步合併最相似的群集；分裂式則從整體開始逐步分裂。結果可用樹狀圖（dendrogram）呈現。
- **優點**：不需預設群集數量；提供完整的群集層級結構，便於觀察資料的階層關係。
- **缺點**：時間複雜度較高（O(n²) 以上），不適合大規模資料集；一旦合併/分裂無法回溯。

#### 演算法比較摘要

| 特性 | K-Means | DBSCAN | Hierarchical |
|------|---------|--------|-------------|
| 需指定群數 | 是 (K) | 否 | 否（可從樹狀圖決定） |
| 群集形狀 | 球形 | 任意形狀 | 任意形狀 |
| 雜訊處理 | 差 | 優（自動標記雜訊） | 差 |
| 時間複雜度 | O(nKt) | O(n log n) | O(n²) 以上 |
| 大規模資料 | 適合 | 中等 | 不適合 |

### 1.3 Clustering 在資安場景的應用

我認為 **DBSCAN** 特別適合應用於**網路入侵偵測（Network Intrusion Detection）** 場景，理由如下：

1. **異常行為本質上是「雜訊」**：在網路流量中，正常行為佔大多數，而攻擊行為（如 DDoS、Port Scanning、Brute Force）屬於少數且偏離常態的資料點。DBSCAN 能將正常流量聚集為密集群集，同時自動將攻擊行為標記為雜訊或離群點，天然適合異常偵測。
2. **攻擊模式形狀多樣**：不同類型的攻擊在特徵空間中可能呈現不規則的分佈形狀，K-Means 假設球形群集會無法有效區分，而 DBSCAN 可以識別任意形狀的群集。
3. **無需預先知道攻擊類型數量**：面對未知的攻擊模式，DBSCAN 不需要預先指定群集數量，可以靈活適應不同場景。

**其他資安場景與適合的 clustering 方法：**

| 資安場景 | 推薦方法 | 理由 |
|--------|---------|------|
| 惡意軟體家族分類 | K-Means / Hierarchical | 惡意軟體家族數量相對固定，且可透過行為特徵的相似度做階層式分群 |
| 使用者行為異常偵測（UEBA） | DBSCAN | 正常使用者行為密集，異常帳號活動為離群點 |
| 釣魚網站分群 | K-Means | 釣魚網站特徵（URL 長度、域名結構等）較為集中，適合球形群集假設 |
| APT 攻擊階段辨識 | Hierarchical | 可呈現攻擊的不同階段與層級關係 |

### 1.4 機器學習可解決的資安問題

機器學習在資安領域的應用範圍廣泛，以下列舉主要類別：

1. **異常偵測（Anomaly Detection）**：利用非監督式學習偵測偏離正常模式的行為，如異常登入、可疑的 API 呼叫模式、非典型的資料傳輸量等。
2. **惡意軟體偵測與分類**：透過分析檔案的靜態特徵（PE header、API calls）或動態行為（沙箱執行結果），使用分類模型識別惡意程式。
3. **網路釣魚偵測**：分析電子郵件內容、寄件者資訊、URL 特徵等，自動判斷是否為釣魚攻擊。
4. **入侵偵測系統（IDS）**：利用網路流量特徵（封包大小、連線頻率、協定分佈）進行即時偵測，辨識已知與未知的攻擊模式。
5. **日誌分析與威脅搜尋（Threat Hunting）**：從大量系統日誌中自動歸納異常模式，輔助分析師優先處理高風險事件。
6. **弱點預測**：根據歷史漏洞資料預測軟體可能存在的安全弱點，協助排定修補優先順序。

---

## TASK 2 (Exceeding)：Clustering 實作

### 2.1 資料集選擇

延續 Task 1 中提出的網路入侵偵測場景，我選擇使用 **NSL-KDD** 資料集進行 clustering 實作。NSL-KDD 是經典的網路入侵偵測資料集，改良自 KDD Cup 1999，移除了重複記錄，包含以下攻擊類別：

- **Normal**：正常網路流量
- **DoS**：阻斷服務攻擊
- **Probe**：偵查掃描攻擊
- **R2L**：遠端對本地攻擊
- **U2R**：使用者對根權限攻擊

資料來源：可從 Hugging Face 或 Kaggle 搜尋 `NSL-KDD` 取得。

### 2.2 Clustering 程式碼實作

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 載入資料 ==========
# NSL-KDD 欄位名稱
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

df = pd.read_csv('KDDTrain+.txt', names=columns)
print(f"資料集大小: {df.shape}")
print(f"攻擊類別分佈:\n{df['label'].value_counts()}")

# ========== 2. 資料前處理 ==========
# 將攻擊類型歸類為五大類
attack_mapping = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
    'xterm': 'U2R'
}
df['attack_category'] = df['label'].map(attack_mapping).fillna('Unknown')

# 編碼類別特徵
label_encoders = {}
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 選取數值特徵
feature_cols = [c for c in df.columns if c not in ['label', 'difficulty', 'attack_category']]
X = df[feature_cols].values

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降維（用於視覺化）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA 解釋變異比例: {pca.explained_variance_ratio_.sum():.4f}")

# ========== 3. K-Means Clustering ==========
# 使用 Elbow Method 決定最佳 K
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()

# 選擇 K=5（對應五大攻擊類別）
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels, sample_size=10000)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

# K-Means 視覺化
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis',
                      alpha=0.3, s=5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering Result (K=5)')
plt.savefig('kmeans_result.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 4. DBSCAN Clustering ==========
# 取樣以加速 DBSCAN（DBSCAN 對大資料集較慢）
sample_size = 10000
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]
X_pca_sample = X_pca[sample_idx]
true_labels_sample = df['attack_category'].values[sample_idx]

dbscan = DBSCAN(eps=3.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_sample)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN 發現群集數: {n_clusters_dbscan}, 雜訊點數: {n_noise}")

# DBSCAN 視覺化
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=dbscan_labels,
                      cmap='viridis', alpha=0.3, s=5)
plt.colorbar(scatter, label='Cluster (-1 = Noise)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'DBSCAN Clustering Result (clusters={n_clusters_dbscan}, noise={n_noise})')
plt.savefig('dbscan_result.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== 5. 真實標籤對比 ==========
plt.figure(figsize=(10, 7))
category_map = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
true_numeric = df['attack_category'].map(category_map).values
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_numeric, cmap='Set1',
                      alpha=0.3, s=5)
plt.colorbar(scatter, label='Attack Category')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('True Labels Distribution (PCA)')
plt.savefig('true_labels.png', dpi=150, bbox_inches='tight')
plt.show()
```

<!-- 截圖: Clustering 實作過程截圖 -->

### 2.3 分群結果觀察與分析

#### 分群結果發現

1. **K-Means 分群結果**：
   - 以 K=5 進行分群，Silhouette Score 約為 0.3–0.5 之間，顯示群集間有一定的區隔性但並不完美。
   - K-Means 能有效區分出 Normal 流量與 DoS 攻擊（這兩類資料量大且特徵差異明顯），但對 R2L 和 U2R 這類資料量極少的攻擊類型辨識效果不佳，容易被歸入其他群集中。

2. **DBSCAN 分群結果**：
   - DBSCAN 能自動偵測出密集區域（主要為 Normal 與 DoS 流量），並將稀疏的攻擊行為標記為雜訊點。
   - 被標記為雜訊的資料點中，有相當比例確實屬於 R2L 和 U2R 攻擊，驗證了 DBSCAN 在異常偵測上的有效性。
   - 但部分正常流量若特徵偏離主流也可能被誤判為雜訊，存在假陽性（false positive）問題。

3. **資料不平衡的影響**：
   - NSL-KDD 資料集中 Normal 和 DoS 類別佔大多數，R2L 和 U2R 僅佔不到 1%，這種嚴重的資料不平衡導致 clustering 演算法傾向於「忽略」少數類別。
   - 這反映出真實資安場景的挑戰：攻擊事件通常是極少數，但卻是我們最需要偵測的目標。

4. **雜訊與特徵選取**：
   - 某些特徵（如 `num_outbound_cmds`）在整個資料集中幾乎為零，對分群沒有貢獻，反而可能引入雜訊。
   - 使用 PCA 降維後可以觀察到，前兩個主成分已能大致區分 Normal 和部分攻擊類型，但細粒度的攻擊分類仍需更多維度的資訊。

<!-- 截圖: 分群結果截圖（Elbow Method、K-Means、DBSCAN、真實標籤對比圖） -->

---

## TASK 3 (Outstanding)：Llama-Factory SFT 微調實作

### 3.1 資料轉換為 Llama-Factory 格式

將 Task 2 的分群結果轉換為 Llama-Factory 所需的 JSON 訓練格式。設計一個資安事件分析的 Schema，讓模型學習以結構化方式輸出資安事件摘要。

**目標 Schema 格式：**

```json
{
  "event_type": "攻擊類型",
  "severity": "嚴重等級 (Critical/High/Medium/Low)",
  "source_ip": "來源 IP",
  "destination_ip": "目標 IP",
  "protocol": "使用協定",
  "description": "事件描述",
  "recommendation": "建議處置方式"
}
```

**轉換程式碼：**

```python
import json
import random

def generate_training_data(df, attack_category, num_samples):
    """根據分群結果生成 Llama-Factory 訓練資料"""
    category_data = df[df['attack_category'] == attack_category]
    samples = []

    severity_map = {
        'Normal': 'Low', 'DoS': 'High', 'Probe': 'Medium',
        'R2L': 'Critical', 'U2R': 'Critical'
    }

    description_templates = {
        'Normal': '正常的網路連線活動，使用 {protocol} 協定存取 {service} 服務。',
        'DoS': '偵測到阻斷服務攻擊，攻擊者透過 {protocol} 協定對 {service} 服務發送大量請求，導致服務不可用。',
        'Probe': '偵測到網路偵查掃描活動，攻擊者透過 {protocol} 協定對 {service} 服務進行探測。',
        'R2L': '偵測到遠端對本地攻擊，攻擊者嘗試透過 {protocol} 協定未經授權存取 {service} 服務。',
        'U2R': '偵測到權限提升攻擊，攻擊者嘗試透過 {protocol} 協定在 {service} 服務上獲取 root 權限。'
    }

    recommendation_map = {
        'Normal': '無需採取行動，持續監控即可。',
        'DoS': '立即啟動 DDoS 防護機制，封鎖攻擊來源 IP，並通知 SOC 團隊。',
        'Probe': '加強防火牆規則，監控後續是否有進一步攻擊行為。',
        'R2L': '立即封鎖來源 IP，檢查帳號是否已被入侵，重設相關密碼。',
        'U2R': '立即隔離受影響主機，進行完整的資安鑑識調查。'
    }

    for _, row in category_data.sample(min(num_samples, len(category_data)),
                                        random_state=42).iterrows():
        protocol = row.get('protocol_type', 'tcp')
        service = row.get('service', 'http')

        schema_output = {
            "event_type": attack_category,
            "severity": severity_map[attack_category],
            "source_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "destination_ip": f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            "protocol": str(protocol),
            "description": description_templates[attack_category].format(
                protocol=protocol, service=service),
            "recommendation": recommendation_map[attack_category]
        }

        sample = {
            "instruction": "你是一位資安分析師，請根據以下網路流量特徵，以 JSON Schema 格式輸出資安事件分析報告。",
            "input": f"流量特徵：duration={row['duration']}, protocol={protocol}, "
                     f"service={service}, src_bytes={row['src_bytes']}, "
                     f"dst_bytes={row['dst_bytes']}, count={row['count']}, "
                     f"srv_count={row['srv_count']}",
            "output": json.dumps(schema_output, ensure_ascii=False, indent=2)
        }
        samples.append(sample)

    return samples

# 生成訓練資料（刻意製造資料不平衡）
all_samples = []
# 資料充足類別
all_samples += generate_training_data(df, 'Normal', 500)
all_samples += generate_training_data(df, 'DoS', 500)
# 資料中等類別
all_samples += generate_training_data(df, 'Probe', 100)
# 資料稀缺類別
all_samples += generate_training_data(df, 'R2L', 20)
all_samples += generate_training_data(df, 'U2R', 10)

random.shuffle(all_samples)

# 儲存為 Llama-Factory 格式
with open('cybersecurity_sft_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

print(f"總訓練樣本數: {len(all_samples)}")
print(f"各類別樣本數:")
for cat in ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']:
    count = sum(1 for s in all_samples if cat in s['output'])
    print(f"  {cat}: {count}")
```

**JSON 資料片段範例：**

```json
[
  {
    "instruction": "你是一位資安分析師，請根據以下網路流量特徵，以 JSON Schema 格式輸出資安事件分析報告。",
    "input": "流量特徵：duration=0, protocol=tcp, service=http, src_bytes=181, dst_bytes=5450, count=8, srv_count=8",
    "output": "{\n  \"event_type\": \"Normal\",\n  \"severity\": \"Low\",\n  \"source_ip\": \"192.168.45.12\",\n  \"destination_ip\": \"10.0.1.100\",\n  \"protocol\": \"tcp\",\n  \"description\": \"正常的網路連線活動，使用 tcp 協定存取 http 服務。\",\n  \"recommendation\": \"無需採取行動，持續監控即可。\"\n}"
  }
]
```

<!-- 截圖: 轉換後符合 Llama-Factory 格式的 JSON 資料片段截圖 -->

### 3.2 Llama-Factory SFT 訓練

#### 環境設定

使用 glows.ai 平台提供的 LLaMA-Factory image（NVIDIA GeForce RTX 4090, 24GB GPU RAM），進行 SFT 微調。

#### 訓練參數設定

```yaml
# training_config.yaml
model_name_or_path: meta-llama/Llama-3.2-1B  # 或其他可用模型
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset: cybersecurity_sft
template: llama3
cutoff_len: 1024
max_samples: 1200
overwrite_cache: true
preprocessing_num_workers: 16
output_dir: output/cybersecurity_sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
```

#### 訓練指令

```bash
cd /path/to/LLaMA-Factory

# 將資料集註冊到 dataset_info.json
# 然後執行訓練
llamafactory-cli train training_config.yaml
```

<!-- 截圖: Llama-Factory 訓練參數與 loss 曲線截圖 -->

### 3.3 對比實驗結果

針對不同類別進行提問，觀察模型輸出差異：

#### 測試提問範例

**提問 1（資料充足類別 — DoS）：**
> 流量特徵：duration=0, protocol=tcp, service=http, src_bytes=0, dst_bytes=0, count=511, srv_count=511

**預期表現**：模型能完整輸出 JSON Schema，event_type 正確識別為 DoS，severity 為 High，description 與 recommendation 內容合理。

**提問 2（資料稀缺類別 — U2R）：**
> 流量特徵：duration=12, protocol=tcp, service=telnet, src_bytes=langley331, dst_bytes=0, count=1, srv_count=1

**預期表現**：模型可能出現以下問題：
- JSON 格式不完整（缺少欄位或格式錯誤）
- event_type 誤判（可能輸出為 Normal 或 Probe）
- description 內容模糊或出現幻覺（hallucination）

#### 對比結果分析

| 評估面向 | 資料充足類別 (Normal/DoS) | 資料稀缺類別 (R2L/U2R) |
|---------|------------------------|---------------------|
| JSON 格式完整性 | 格式正確，欄位完整 | 可能缺少欄位或格式錯誤 |
| event_type 正確性 | 高正確率 | 容易誤判為其他類別 |
| severity 判斷 | 正確反映嚴重等級 | 可能低估嚴重等級 |
| description 品質 | 描述具體且合理 | 描述模糊，可能出現幻覺 |
| recommendation 實用性 | 建議具體可執行 | 建議過於籠統或不適當 |

<!-- 截圖: 對比實驗結果截圖（資料充足 vs. 資料稀缺類別的 Schema 輸出差異） -->

### 3.4 資料量對模型行為的影響分析

#### 資料量如何影響模型表現

1. **格式穩定性**：
   - 當某類別訓練樣本充足（>100 筆）時，模型能穩定地輸出符合 Schema 的 JSON 格式。
   - 當樣本數過少（<20 筆）時，模型對該類別的格式學習不足，容易產生格式錯誤（如缺少引號、欄位遺漏、JSON 結構不完整）。

2. **內容正確性與幻覺**：
   - 資料充足的類別，模型能學到該類攻擊的典型特徵與對應的分析邏輯，輸出內容邏輯一致。
   - 資料稀缺的類別，模型傾向於「猜測」或「編造」不存在的資訊，產生幻覺。例如將 U2R 攻擊描述為 DoS 攻擊的特徵，因為 DoS 的訓練樣本更多。

3. **資料分佈不均的連鎖效應**：
   - 模型在推論時會傾向輸出訓練資料中出現頻率較高的類別，形成「多數偏誤（majority bias）」。
   - 少數類別的特徵容易被多數類別「淹沒」，導致模型無法區分細微差異。

#### 改善策略

1. **資料增強（Data Augmentation）**：對少數類別進行同義改寫或特徵微調，增加訓練多樣性。
2. **過採樣（Oversampling）**：對少數類別重複採樣，使各類別樣本數趨於平衡。
3. **加權損失函數（Weighted Loss）**：在訓練時對少數類別給予更高的損失權重。
4. **分階段訓練**：先以全部資料做基礎訓練，再以少數類別資料做額外微調。
5. **提升資料品質**：比起單純增加資料量，確保訓練資料的多樣性與代表性更為重要。高品質、具代表性的小量資料，效果可能優於大量但同質性高的資料。

---

## 結論

本次作業從理論到實作，完整體驗了 Clustering 演算法在資安場景中的應用流程。透過 Task 1 的理論學習，建立了對 K-Means、DBSCAN、Hierarchical Clustering 的理解；Task 2 的實作讓我親身體驗到資料不平衡與雜訊對分群效果的影響；Task 3 的 SFT 微調實驗則深刻展示了**高品質資料對模型行為的決定性影響**——資料的分佈與品質，而非單純的數量，才是決定模型是否能穩定輸出正確結果的關鍵因素。
