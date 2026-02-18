# BERT 模型离线使用说明

## 问题
代码尝试从 Hugging Face 下载 BERT 模型，但网络连接可能有问题。

## 解决方案

### 方法 1: 使用下载脚本（如果有网络）
```bash
cd /root/LIBERO
eval "$(conda shell.bash hook)"
conda activate libero
python download_bert_model.py
```

### 方法 2: 手动下载模型文件

1. **创建目录**：
```bash
mkdir -p /root/LIBERO/bert/bert-base-cased
```

2. **下载以下文件到 `/root/LIBERO/bert/bert-base-cased/` 目录**：
   - `config.json`
   - `pytorch_model.bin` (或 `model.safetensors`)
   - `tokenizer_config.json`
   - `vocab.txt`
   - `tokenizer.json`

3. **下载地址**：
   - 主页面：https://huggingface.co/bert-base-cased
   - 直接下载链接：
     - https://huggingface.co/bert-base-cased/resolve/main/config.json
     - https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
     - https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json
     - https://huggingface.co/bert-base-cased/resolve/main/vocab.txt
     - https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json

4. **使用 wget 下载**（如果有网络）：
```bash
cd /root/LIBERO/bert/bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/config.json
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json
```

## 验证
下载完成后，运行以下命令验证：
```bash
ls -lh /root/LIBERO/bert/bert-base-cased/
```

应该看到所有必需的文件。
