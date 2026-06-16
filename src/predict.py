import torch
import tiktoken
import config
from model import GPT


def get_latest_checkpoint():
    checkpoint_files = sorted(config.MODELS_DIR.glob("model_epoch*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"在 {config.MODELS_DIR} 下没有找到模型检查点")
    return checkpoint_files[-1]

def load_model(device):
    checkpoint_path = get_latest_checkpoint()
    model = GPT(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ===== 核心修复：去掉 torch.compile 自动加的 _orig_mod. 前缀 =====
    raw_state_dict = checkpoint["model_state_dict"]
    fixed_state_dict = {}
    for key, value in raw_state_dict.items():
        # 移除前缀 _orig_mod.
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
            
            
    #model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(fixed_state_dict)
    model.eval()

    print(f"模型加载成功: {checkpoint_path}")
    return model

def predict_batch(model, input_ids, max_new_tokens=100):
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return output_ids


def predict(text, model, tokenizer, device, max_new_tokens=100):
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    output_tensor = predict_batch(
        model=model,
        input_ids=input_tensor,
        max_new_tokens=max_new_tokens,
    )

    new_tokens = output_tensor[0, input_tensor.shape[1]:].tolist()
    return tokenizer.decode(new_tokens)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model(device)

    print("欢迎使用 miniGPT2 预测程序")
    print("输入 quit 退出")

    while True:
        user_input = input("请输入文本: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            print("输入不能为空")
            continue

        result = predict(
            text=user_input,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=100,
        )
        print("生成结果:", result)


if __name__ == "__main__":
    run()