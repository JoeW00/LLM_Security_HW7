# Assignment 7：Clustering 演算法與資安應用

> **課程**：LLM Security System（114 學年度第二學期）  
> **學號**：112003854

---

## Task 1 (Achieving)：Clustering 原理與資安場景思考

完成 IBM SkillBuild「Machine Learning for Developers」學習路徑，涵蓋四個實作單元，並撰寫 Clustering 演算法（K-Means、DBSCAN、Hierarchical）的原理比較與資安場景應用分析。

| 檔案 | 說明 |
|------|------|
| `Task1_IBM_first_ML_model.ipynb` | IBM 課程實作 — Random Forest 預測客戶流失 |
| `Task1_IBM_regression.ipynb` | IBM 課程實作 — 5 種迴歸演算法比較 |
| `Task1_IBM_classification.ipynb` | IBM 課程實作 — 5 種分類演算法比較 |
| `Task1_IBM_clustering.ipynb` | IBM 課程實作 — 4 種分群演算法比較 |

---

## Task 2 (Exceeding)：NSL-KDD 網路入侵偵測 Clustering 實作

使用 NSL-KDD 資料集，以 K-Means 與 DBSCAN 對網路流量進行分群，觀察兩種演算法在區分正常連線與攻擊行為（Normal / DoS / Probe / R2L / U2R）上的表現差異。

| 檔案 | 說明 |
|------|------|
| `Task2_NSL_KDD_clustering.ipynb` | Clustering 完整實作（前處理、K-Means、DBSCAN、視覺化、分析） |

---

## Task 3 (Outstanding)：Llama-Factory SFT 微調與資料不平衡實驗

將 Task 2 的分群結果轉換為 SFT 訓練資料，刻意製造類別不平衡（DoS 500 筆 vs U2R 10 筆），以 LoRA 微調 Llama-3.2-1B，觀察資料量對模型分類能力與幻覺行為的影響。

| 檔案 | 說明 |
|------|------|
| `Task3_SFT_data_conversion.ipynb` | 資料轉換程式（NSL-KDD → ShareGPT 格式） |
| `cybersecurity_sft_data.json` | 轉換後的 SFT 訓練資料（1,130 筆） |
| `cyber_sft.yaml` | LlamaFactory LoRA SFT 訓練設定檔 |
