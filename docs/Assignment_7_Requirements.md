# Assignment 7 作業要求拆解

## 主題：Clustering 演算法與資安應用

---

## TASK 1 (Achieving)：Clustering 原理與資安場景思考

> 這是**文字撰寫**為主的任務，不需要寫程式。

### 需要完成的項目：

1. **1.1 IBM SkillBuild 課程學習心得**
   - 完成 IBM SkillBuild「Machine Learning for Developers」學習路徑
   - 撰寫學習心得
   - 附上課程實作過程截圖

2. **1.2 Clustering 演算法理解**
   - 說明三種 clustering 演算法的原理、優缺點：
     - K-Means
     - DBSCAN
     - Hierarchical Clustering
   - 製作演算法比較表（需指定群數、群集形狀、雜訊處理、時間複雜度、大規模資料適用性）

3. **1.3 Clustering 在資安場景的應用**
   - 選擇一種 clustering 方法，說明適合哪個資安場景及理由
   - 列出其他資安場景與推薦方法的對照表

4. **1.4 機器學習可解決的資安問題**
   - 列舉機器學習在資安領域的主要應用類別（異常偵測、惡意軟體、釣魚偵測、IDS、日誌分析、弱點預測等）

### 交付物：
- 文字說明 + 表格 + IBM SkillBuild 截圖

---

## TASK 2 (Exceeding)：Clustering 實作

> 這是**程式實作**任務，需要寫 Python 程式碼並產出視覺化結果。

### 需要完成的項目：

1. **2.1 資料集選擇**
   - 從 Hugging Face 或 Kaggle 取得資料集（作業範例使用 NSL-KDD）
   - 說明資料集內容與攻擊類別（Normal、DoS、Probe、R2L、U2R）

2. **2.2 Clustering 程式碼實作**
   - 資料前處理：
     - 攻擊類型歸類（細分類 → 五大類）
     - 類別特徵編碼（LabelEncoder）
     - 特徵標準化（StandardScaler）
     - PCA 降維（用於視覺化）
   - K-Means Clustering：
     - Elbow Method 決定最佳 K 值
     - 執行 K-Means 分群（K=5）
     - 計算 Silhouette Score
     - 視覺化分群結果
   - DBSCAN Clustering：
     - 設定 eps 和 min_samples 參數
     - 執行 DBSCAN 分群
     - 統計群集數與雜訊點數
     - 視覺化分群結果
   - 真實標籤對比圖

3. **2.3 分群結果觀察與分析**
   - 分析 K-Means 分群結果的表現
   - 分析 DBSCAN 分群結果的表現
   - 討論資料不平衡的影響
   - 討論雜訊與特徵選取

### 交付物：
- Python 程式碼（建議用 Jupyter Notebook）
- 4 張圖片：Elbow Method、K-Means 結果、DBSCAN 結果、真實標籤對比
- 文字分析說明
- 實作過程截圖

---

## TASK 3 (Outstanding)：Llama-Factory SFT 微調實作

> 這是**進階實作**任務，需要進行 LLM 微調訓練。

### 需要完成的項目：

1. **3.1 資料轉換為 Llama-Factory 格式**
   - 設計資安事件分析的 JSON Schema（包含 event_type、severity、source_ip、destination_ip、protocol、description、recommendation）
   - 撰寫轉換程式，將 Task 2 的分群結果轉為訓練資料
   - **刻意製造資料不平衡**：
     - Normal / DoS：各 500 筆
     - Probe：100 筆
     - R2L：20 筆
     - U2R：10 筆
   - 輸出 `cybersecurity_sft_data.json`
   - 附上轉換後的 JSON 資料片段截圖

2. **3.2 Llama-Factory SFT 訓練**
   - 使用 glows.ai 平台（RTX 4090, 24GB GPU）
   - 模型：meta-llama/Llama-3.2-1B（或其他可用模型）
   - 訓練方式：LoRA SFT
   - 設定訓練參數（batch_size、learning_rate、epochs 等）
   - 執行訓練並記錄 loss 曲線
   - 附上訓練參數與 loss 曲線截圖

3. **3.3 對比實驗結果**
   - 對資料充足類別（DoS）進行提問，觀察模型輸出
   - 對資料稀缺類別（U2R）進行提問，觀察模型輸出
   - 比較兩者在以下面向的差異：
     - JSON 格式完整性
     - event_type 正確性
     - severity 判斷
     - description 品質
     - recommendation 實用性
   - 附上對比實驗結果截圖

4. **3.4 資料量對模型行為的影響分析**
   - 分析格式穩定性與資料量的關係
   - 分析內容正確性與幻覺問題
   - 分析資料分佈不均的連鎖效應（多數偏誤）
   - 提出改善策略（資料增強、過採樣、加權損失、分階段訓練、提升資料品質）

### 交付物：
- 資料轉換 Python 程式碼
- `cybersecurity_sft_data.json` 訓練資料檔
- Llama-Factory 訓練設定檔（YAML）
- 訓練 loss 曲線截圖
- 對比實驗結果截圖
- 文字分析說明

---

## 整體難度對照

| Task | 等級 | 性質 | 核心重點 |
|------|------|------|---------|
| Task 1 | Achieving（基礎） | 文字撰寫 | 理論理解 + 資安場景思考 |
| Task 2 | Exceeding（進階） | 程式實作 | Clustering 實作 + 結果分析 |
| Task 3 | Outstanding（卓越） | LLM 微調 | SFT 訓練 + 資料不平衡影響 |

## 需要的工具與環境

- **Task 1**：無特殊需求，文字編輯即可
- **Task 2**：Python + scikit-learn + matplotlib + pandas + numpy
- **Task 3**：glows.ai 平台 + Llama-Factory + GPU (RTX 4090)
