import os
import torch
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class EarlyStopping:
    """
    早停类：当验证集准确率在一定轮数内不再提高时，停止训练
    
    参数:
        patience: 容忍验证集准确率不提升的轮数
        delta: 判定准确率是否提升的阈值
        verbose: 是否打印早停信息
    """
    def __init__(self, patience=5, delta=0, verbose=True):
        # patience: 容忍多少个epoch验证集准确率不提升
        self.patience = patience 
        # delta: 判定准确率提升的最小变化阈值,小于此值视为没有提升
        self.delta = delta 
        # counter: 记录连续多少个epoch验证准确率没有提升
        self.counter = 0 
        # best_score: 记录历史最佳验证准确率
        self.best_score = None
        # early_stop: 是否触发早停的标志
        self.early_stop = False
        
    def __call__(self, val_acc):
        # 获取当前epoch的验证准确率
        score = val_acc
        
        # 如果是第一次调用,初始化best_score为当前准确率
        if self.best_score is None:
            self.best_score = score
            return
        
        # 当前准确率未超过历史最佳准确率+阈值delta时
        if score <= self.best_score + self.delta:
            # 累加未提升的epoch计数
            self.counter += 1
            # 当未提升的epoch数达到patience时触发早停
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"早停触发! 最佳验证准确率: {self.best_score:.4f}")
        # 当前准确率超过历史最佳时
        else:
            # 更新历史最佳准确率
            self.best_score = score
            # 重置未提升的epoch计数
            self.counter = 0


class ModelSaver:
    """
    模型保存类：根据配置保存模型权重
    
    参数:
        save_dir: 模型保存目录
        save_best_only: 是否只保存最佳模型
        verbose: 是否打印保存信息
    """
    def __init__(self, save_dir='model_weights', save_best_only=True):
        # 初始化保存路径，默认为'model_weights'
        self.save_dir = save_dir
        # 是否只保存最佳模型的标志，默认为True
        self.save_best_only = save_best_only
        # 记录最佳验证准确率，初始为None
        self.best_score = None
        
        # 如果保存目录不存在，则创建该目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def __call__(self, model, epoch, val_acc):
        """
        保存模型
        
        参数:
            model: 需要保存的模型
            epoch: 当前训练轮数
            val_acc: 当前验证准确率
        """
        # 根据epoch和准确率生成模型文件名
        filename = f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth'
        save_path = os.path.join(self.save_dir, filename)
        
        if self.save_best_only:
            # 如果是首次保存，或当前模型性能优于之前最佳模型
            if self.best_score is None or val_acc > self.best_score:
                # 更新最佳分数
                self.best_score = val_acc
                # 保存当前模型
                torch.save(model.state_dict(), save_path)
                
                # 删除之前保存的其他模型文件
                for old_file in os.listdir(self.save_dir):
                    if old_file != filename and old_file.endswith('.pth'):
                        os.remove(os.path.join(self.save_dir, old_file))
        else:
            # 如果不是只保存最佳模型，则每个epoch都保存
            torch.save(model.state_dict(), save_path)


def evaluate_classification_model(model, test_loader, device, criterion=None):
    """评估模型在测试集上的性能
    
    参数:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 运行设备(GPU/CPU)
        criterion: 损失函数,默认为None
        
    返回:
        accuracy: 模型在测试集上的准确率
        avg_loss: 如果提供了criterion,则返回平均损失值
    """
    # 将模型设置为评估模式,不进行反向传播
    model.eval()  # 设置为评估模式
    # 初始化正确预测数和总样本数的计数器
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():  # 不计算梯度
        # 遍历测试数据集中的每一批数据
        for images, labels in test_loader:
            # 将数据转移到指定设备(GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            # 前向传播得到模型输出
            outputs = model(images)
            # 获取每个样本的预测结果(最大概率的类别)
            _, predicted = torch.max(outputs.data, 1)
            # 累加当前批次的样本总数
            total += labels.size(0)
            # 累加当前批次中预测正确的样本数
            correct += (predicted == labels).sum().item()
            
            # 如果提供了损失函数，计算损失
            if criterion is not None:
                # 计算当前批次的损失值
                loss = criterion(outputs, labels)
                # 累加损失值(乘以批次大小以获得加权和)
                total_loss += loss.item() * labels.size(0)
    
    # 计算总体准确率(百分比)
    accuracy = 100 * correct / total
    
    if criterion is not None:
        # 计算平均损失(总损失除以总样本数)
        avg_loss = total_loss / total
        return accuracy, avg_loss
    
    return accuracy



def train_classification_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs, 
    tensorboard_logger=None,
    model_saver=None,
    early_stopping=None,
    eval_step=500
):
    """
    基于tqdm的训练函数，与training函数类似
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        tensorboard_callback: Tensorboard回调函数
        model_saver: 保存检查点回调函数
        early_stopping: 早停回调函数
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    # 初始化记录字典
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    
    # 创建进度条（使用notebook兼容版本）
    pbar = tqdm(total=num_epochs * len(train_loader), desc="Training Progress")
    
    try:
        for epoch_id in range(num_epochs):
            for datas, labels in train_loader:
                datas = datas.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = model(datas)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                preds = logits.argmax(axis=-1)
                acc = (preds == labels).float().mean().item() * 100
                loss_value = loss.item()
                
                record_dict["train"].append({
                    "loss": loss_value, 
                    "acc": acc, 
                    "step": global_step
                })
                
                if global_step % eval_step == 0:
                    val_acc, val_loss = evaluate_classification_model(model, val_loader, device, criterion)
                    record_dict["val"].append({
                        "loss": val_loss, 
                        "acc": val_acc, 
                        "step": global_step
                    })
                    model.train()
                    
                    if tensorboard_logger:
                        tensorboard_logger.log_training(global_step, loss_value, acc)
                        tensorboard_logger.log_validation(global_step, val_loss, val_acc)
                    
                    if model_saver:
                        model_saver(model, epoch_id, val_acc)
                    
                    if early_stopping:
                        early_stopping(val_acc)
                        if early_stopping.early_stop:
                            pbar.close()
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model, record_dict
                
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": epoch_id, 
                    "loss": f"{loss_value:.4f}", 
                    "acc": f"{acc:.2f}%",
                    "val_acc": f"{val_acc:.2f}%" if global_step % eval_step == 0 else "N/A"
                })
    
    finally:
        pbar.close()
    
    return model, record_dict


def plot_learning_curves(record_dict, sample_step=500):
    """
    画学习曲线，横坐标是steps，纵坐标是loss和acc,回归问题只有loss
    
    参数:
        record_dict: 包含训练和验证记录的字典
        sample_step: 每多少步画一个点，默认500步
    """
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    fig_num = len(train_df.columns)  # 因为有loss和acc两个指标，所以画个子图
    fig, axs = plt.subplots(1, fig_num, figsize=(5 * fig_num, 5))  # fig_num个子图，figsize是子图大小
    for idx, item in enumerate(train_df.columns):
        # index是步数，item是指标名字
        axs[idx].plot(train_df.index, train_df[item], label=f"train_{item}")
        axs[idx].plot(val_df.index, val_df[item], label=f"val_{item}")
        axs[idx].grid()
        axs[idx].legend()
        x_data = range(0, train_df.index[-1], 5000)  # 每隔5000步标出一个点
        axs[idx].set_xticks(x_data)
        axs[idx].set_xticklabels(map(lambda x: f"{int(x / 1000)}k", x_data))  # map生成labal
        axs[idx].set_xlabel("step")

    plt.show()



def plot_learning_loss_curves(record_dict, sample_step=500):
    """
    画学习曲线，横坐标是steps，纵坐标是loss和acc,回归问题只有loss

    参数:
        record_dict: 包含训练和验证记录的字典
        sample_step: 每多少步画一个点，默认500步
    """
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    # 只绘制一个loss图，不需要循环
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建单个图表

    # 绘制训练和验证的loss曲线
    ax.plot(train_df.index, train_df['loss'], label="train_loss")
    ax.plot(val_df.index, val_df['loss'], label="val_loss")

    # 设置图表属性
    ax.grid()
    ax.legend()
    x_data = range(0, train_df.index[-1], 5000)  # 每隔5000步标出一个点
    ax.set_xticks(x_data)
    ax.set_xticklabels(map(lambda x: f"{int(x / 1000)}k", x_data))  # map生成label
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")

    plt.show()


def train_regression_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device='cpu', 
    num_epochs=100, 
    print_every=10,
    eval_step=500,
    model_saver=None,
    early_stopping=None
):
    """
    训练回归模型的函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        print_every: 每多少轮打印一次结果
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss=0
    
    with tqdm(total=num_epochs * len(train_loader), desc="train progress") as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            model.train()
            running_loss = 0.0
            
            for datas,labels in train_loader:
                # 假设inputs是一个包含多个tensor的元组，targets是最后一个元素
                targets = labels.to(device)
                if isinstance(datas,list):#如果datas是tuple，则将每个元素都放到device上
                    inputs = [x.to(device) for x in datas]
                else:
                    inputs = datas.to(device)
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 模型前向计算
                if isinstance(inputs,list):
                    outputs = model(*inputs)  # 使用解包操作将多个输入传递给模型
                else:
                    outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 更新步骤
                global_step += 1
            
                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })
                
                # 验证
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_regression_model(model, val_loader, device, criterion)
                    model.train()
            
                    # 记录验证数据
                    record_dict["val"].append({
                        "loss": epoch_val_loss, 
                        "step": global_step
                    })
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss, epoch_id)
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model,record_dict
            
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{epoch_train_loss:.4f}", "val_loss": f"{epoch_val_loss:.4f},global_step{global_step}"})
    
    return model, record_dict



def evaluate_regression_model(model, dataloader,  device,criterion):
    model.eval()
    running_loss = 0.0

    
    with torch.no_grad():#禁止 autograd 记录计算图，节省显存与算力。
        for inputs, targets in dataloader:
            targets = targets.to(device)
            if isinstance(inputs, list):  # 如果datas是tuple，则将每个元素都放到device上
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)

            # 模型前向计算
            if isinstance(inputs, list):
                outputs = model(*inputs)  # 使用解包操作将多个输入传递给模型
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets) #计算损失
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)



def train_multi_output_regression_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device='cpu', 
    num_epochs=100, 
    print_every=10,
    eval_step=500,
    model_saver=None,
    early_stopping=None
):
    """
    训练回归模型的函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        print_every: 每多少轮打印一次结果
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss=0
    
    with tqdm(total=num_epochs * len(train_loader), desc="train progress") as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            model.train()
            running_loss = 0.0
            
            for datas,labels in train_loader:
                # 假设inputs是一个包含多个tensor的元组，targets是最后一个元素
                targets = labels.to(device)
                if isinstance(datas,list):#如果datas是tuple，则将每个元素都放到device上
                    inputs = [x.to(device) for x in datas]
                else:
                    inputs = datas.to(device)
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 模型前向计算
                if isinstance(inputs,list):
                    outputs = model(*inputs)  # 使用解包操作将多个输入传递给模型
                else:
                    outputs = model(inputs)
                # 计算损失
                output,deep=outputs
                # 处理deep：求平均，reshape为output尺寸，并和output相加
                deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
                deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
                # 分别计算output和deep_reshaped的损失，然后求和
                loss_output = criterion(output, targets)
                loss_deep = criterion(deep_reshaped, targets)
                loss = loss_output + loss_deep  # 总损失为两部分之和
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 更新步骤
                global_step += 1
            
                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })
                
                # 验证
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_multi_output_regression_model(model, val_loader, device, criterion)
                    model.train()
            
                    # 记录验证数据
                    record_dict["val"].append({
                        "loss": epoch_val_loss, 
                        "step": global_step
                    })
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss, epoch_id)
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model,record_dict
            
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{epoch_train_loss:.4f}", "val_loss": f"{epoch_val_loss:.4f},global_step{global_step}"})
    
    return model, record_dict


def evaluate_multi_output_regression_model(model, dataloader,  device,criterion):
    model.eval()
    running_loss = 0.0

    
    with torch.no_grad():#禁止 autograd 记录计算图，节省显存与算力。
        for inputs, targets in dataloader:
            targets = targets.to(device)
            if isinstance(inputs, list):  # 如果datas是tuple，则将每个元素都放到device上
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)

            # 模型前向计算
            if isinstance(inputs, list):
                outputs = model(*inputs)  # 使用解包操作将多个输入传递给模型
            else:
                outputs = model(inputs)
            output,deep=outputs
            # 处理deep：求平均，reshape为output尺寸，并和output相加
            deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
            deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
            loss_output = criterion(output, targets)
            loss_deep = criterion(deep_reshaped, targets)
            loss = loss_output + loss_deep  # 总损失为两部分之和
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)