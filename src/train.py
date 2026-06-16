from tqdm import tqdm
import keyboard
import sys
import matplotlib.pyplot as plt
import time
import random
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloader
import config
from model import GPT
from torch.amp import GradScaler, autocast  # type: ignore  #屏蔽静态检查，要不虽然没错但是 Python 解释器会标红
import sys
# 强制行缓冲，实时刷新输出
sys.stdout.reconfigure(line_buffering=True) # type: ignore  #屏蔽静态检查，要不虽然没错但是 Python 解释器会标红

torch.manual_seed(1024)
random.seed(1024)
stop_training_flag = False

#键盘监听，用于中断
def on_key_press(event):
    """键盘回调函数：按下键时触发"""
    global stop_training_flag
    if event.name == config.key:
        print("\n[中断请求] 检测到按键中断。将在当前 Batch 结束后停止训练...")
        stop_training_flag = True
        keyboard.unhook_all() # 取消监听，防止重复触发
        
# 保存模型 
def save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, save_dir, suffix=""):
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"model_epoch_{epoch}{suffix}.pt"
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": avg_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[保存] 模型检查点已保存至: {checkpoint_path}")
    
# 保存训练图片
def save_plots(train_losses, val_losses, batch_losses, save_dir, suffix=""):
    save_dir.mkdir(parents=True, exist_ok=True)

    if train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker="o")
        if val_losses:
            plt.plot(val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / f"loss_curve{suffix}.png", dpi=150)
        plt.close()

    if batch_losses:
        plt.figure(figsize=(12, 5))
        plt.plot(batch_losses, color="g", alpha=0.7)
        plt.xlabel("Batch Index")
        plt.ylabel("Batch Loss")
        plt.title("Training Loss per Batch")
        plt.grid(True)
        plt.savefig(save_dir / f"batch_loss{suffix}.png", dpi=150)
        plt.close()

    print(f"[保存] 曲线图已保存至: {save_dir}")  
    
# 单批次训练函数 
def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, batch_losses,writer, global_step):  
    global stop_training_flag
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True,mininterval=1.0)
    for batch_idx, (x, y) in enumerate(pbar):
        # 监听键盘事件，触发就中断循环
        if stop_training_flag:
            break
        x = x.to(device)
        y = y.to(device)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, loss = model(x, targets=y) 
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        scaler.step(optimizer)  
        scaler.update()
        
        loss_item = loss.item()
        total_loss += loss_item
        batch_losses.append(loss_item)
          
        writer.add_scalar("train/batch_loss", loss_item, global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1
        
        pbar.set_postfix(loss=f"{loss_item:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # if batch_idx % 10 == 0:
        #     print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss_item:.4f}")
            
    avg_loss = total_loss / max(len(train_loader), 1) 
    return avg_loss,global_step
    

# 评估函数
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            _, loss = model(x, targets=y)

        total_loss += loss.item()

    avg_loss = total_loss / max(len(val_loader), 1)
    return avg_loss


def main():
    global stop_training_flag
    stop_training_flag = False
    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    # 2. 数据集
    train_loader = get_dataloader(train=True)
    val_loader = get_dataloader(train=False)
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 3. 模型
    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6} M")

    
    # 4. 优化器
    # 混合精度训练
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 模型编译,性能不好不用开
    print(">>> 正在编译模型 (torch.compile)，第一次运行会慢一点...")
    model = torch.compile(model)
    # 设置学习率调整次数与epoch数量一致
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epoch)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    checkpoint_dir = config.MODELS_DIR
    plot_dir = config.LOGS_DIR
    # 确保文件夹存在（如果不存在就创建）
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # tensorboard 记录数据图像
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))
    global_step = 0 
    
    train_losses = []
    val_losses = []
    batch_losses = []
    
    total_start_time = time.time() #记录时间
    best_val_loss = float('inf')  # 在训练循环外初始化，只保留loss更小的参数
    epochs_no_improve = 0
    patience = config.patience # 早停参数
    num_epochs = config.epoch
    
    # 注册键盘监听
    keyboard.on_press(on_key_press)
    print(">>> 训练已开始，按键盘 'p' 可在当前 batch 结束后中断并保存。<<<")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 外层检查键盘监听
        if stop_training_flag:
            break
        
        print(f"\n--- 开始训练 Epoch {epoch} ---")
        epoch_start_time = time.time()
        
        avg_train_loss,global_step  = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            batch_losses=batch_losses,
            writer=writer,
            global_step=global_step,
        )
        
        avg_val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 写入tensorboard
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
        
        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - total_start_time
        
        print(
            f"Epoch: {epoch}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Epoch Time: {epoch_duration:.2f}s, "
            f"Total Time: {total_duration:.2f}s"
        )
        
        # 保存模型及早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                avg_val_loss=avg_val_loss,
                save_dir=checkpoint_dir,
            )
        else:
            epochs_no_improve += 1
            print(f"Val loss not improved for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("触发早停")
                break
        
        # 再检查键盘监听，最后保存一次模型
        if stop_training_flag:
            print("\n[流程结束] 训练已由用户中断。")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                avg_val_loss=avg_val_loss,
                save_dir=checkpoint_dir,
                suffix="_interrupted",
            )
            save_plots(train_losses, val_losses, batch_losses, plot_dir, suffix="_interrupted")
            break
        
    if not stop_training_flag:
        save_plots(train_losses, val_losses, batch_losses, plot_dir)
        print("\n[流程结束] 训练正常完成。")
    
    writer.close() 
    total_duration = time.time() - total_start_time
    print(f"总共耗时: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    
    try:
        keyboard.unhook_all()
    except Exception:
        pass
    
if __name__ == "__main__":
    main()